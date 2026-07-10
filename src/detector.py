"""
Module: detector.py
Description: Detects and counts objects in images using YOLOv8.
Supports both pre-trained models and custom trained models.
Includes robust NMS post-processing to handle overlapping detections.
"""

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path

from src.shape_analyzer import ShapeAnalyzer

# SAM 2 — optional; degrades gracefully if the package is not installed.
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    _SAM2_AVAILABLE = True
except ImportError:
    _SAM2_AVAILABLE = False
    SAM2ImagePredictor = None


class ObjectDetector:
    """
    YOLOv8-based object detector for counting and classifying objects in images.
    """

    TEMPLATE_WIDTH_MM = 113.52
    TEMPLATE_HEIGHT_MM = 180.41
    # Maximum boxes processed in a single SAM 2 decoder pass.
    # Keeps peak VRAM usage predictable on consumer GPUs (e.g. GTX 1650 4 GB).
    SAM_CHUNK_SIZE = 64
    
    def __init__(self, model_path="yolov8n.pt", use_sam2=False, sam2_device=None):
        """
        Initialize the detector with a YOLOv8 model.
        
        Args:
            model_path (str): Path to the YOLOv8 model file.
            use_sam2 (bool): If True, also load a SAM 2 predictor for high-fidelity
                masks.  Silently disabled when the ``sam2`` package is not installed.
            sam2_device (str | None): Device for SAM 2 inference (``'cuda'``,
                ``'cpu'``, etc.).  Auto-detected when ``None``.
        """
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.shape_analyzer = ShapeAnalyzer()
        print(f"[INFO] Model loaded: {model_path}")

        # SAM 2 predictor (optional)
        self.use_sam2 = use_sam2 and _SAM2_AVAILABLE
        self.sam2_predictor = None
        self.sam2_device = 'cpu'
        if self.use_sam2:
            try:
                if sam2_device is None:
                    sam2_device = 'cuda' if torch.cuda.is_available() else 'cpu'
                _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                _sam2_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml'
                _sam2_weights = os.path.join(_root, 'models', 'sam2.1_hiera_large.pt')
                _sam2_model = build_sam2(_sam2_cfg, _sam2_weights, device=sam2_device)
                self.sam2_predictor = SAM2ImagePredictor(_sam2_model)
                self.sam2_device = sam2_device
                # ── Phase 4: Enable PyTorch SDPA memory-efficient attention ──
                if sam2_device == 'cuda' and torch.cuda.is_available():
                    try:
                        torch.backends.cuda.enable_flash_sdp(True)
                        print('[INFO] SAM 2: Flash-Attention SDP enabled')
                    except Exception:
                        pass
                    try:
                        torch.backends.cuda.enable_mem_efficient_sdp(True)
                        print('[INFO] SAM 2: Memory-efficient SDP enabled')
                    except Exception:
                        pass
                print(f'[INFO] SAM 2 predictor initialized on {sam2_device}')
            except Exception as e:
                print(f'[WARN] SAM 2 initialization failed: {e}. Falling back to OpenCV.')
                self.use_sam2 = False
                self.sam2_predictor = None
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _apply_nms(self, detections, iou_threshold=0.4):
        """
        Apply Non-Maximum Suppression to remove overlapping detections.
        Keeps the detection with highest confidence for overlapping boxes.
        
        Args:
            detections: List of detection dicts with 'box' and 'confidence'
            iou_threshold: Threshold for considering boxes as overlapping
            
        Returns:
            List of filtered detections
        """
        if len(detections) == 0:
            return []
        
        # Sort by confidence (descending)
        sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        keep_indices = []
        
        for i, det in enumerate(sorted_dets):
            keep = True
            for kept_idx in keep_indices:
                iou = self._calculate_iou(det['box'], sorted_dets[kept_idx]['box'])
                if iou > iou_threshold:
                    keep = False
                    break
            
            if keep:
                keep_indices.append(i)
        
        return [sorted_dets[i] for i in keep_indices]

    def _shrink_box(self, box, shrink_ratio=0.15):
        """
        Shrink a bounding box by a percentage to reduce padding.
        
        Args:
            box: [x1, y1, x2, y2] coordinates
            shrink_ratio: Fraction to shrink from each side (0.15 = 15%)
            
        Returns:
            Shrunk box coordinates
        """
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        
        # Shrink from all sides
        shrink_x = int(w * shrink_ratio)
        shrink_y = int(h * shrink_ratio)
        
        x1_new = x1 + shrink_x
        y1_new = y1 + shrink_y
        x2_new = x2 - shrink_x
        y2_new = y2 - shrink_y
        
        # Ensure box is still valid
        if x2_new <= x1_new:
            x1_new = x1
            x2_new = x2
        if y2_new <= y1_new:
            y1_new = y1
            y2_new = y2
        
        return [x1_new, y1_new, x2_new, y2_new]

    def _estimate_length_mm(self, image, box, pixels_per_mm):
        """
        Estimate bean length from a silhouette-derived binary mask.

        This is closer to the bean's central crease than a padded box because
        it measures along the object's main axis instead of the outer rectangle.
        """
        if pixels_per_mm is None:
            return None

        if isinstance(pixels_per_mm, dict):
            ppm_x = float(pixels_per_mm.get('x') or pixels_per_mm.get('avg') or 0)
            ppm_y = float(pixels_per_mm.get('y') or pixels_per_mm.get('avg') or 0)
        else:
            ppm_x = float(pixels_per_mm or 0)
            ppm_y = float(pixels_per_mm or 0)

        if ppm_x <= 0 or ppm_y <= 0:
            return None

        x1, y1, x2, y2 = map(int, box)
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        try:
            _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        except Exception:
            return None

        try:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            contour = self.shape_analyzer._largest_contour(mask)
            if contour is None or len(contour) < 5:
                return None

            analysis = self.shape_analyzer.analyze(mask, ppm_x, ppm_y)
            length_mm = analysis.get('midline_length_mm')
            if length_mm and length_mm > 0:
                return float(length_mm)
        except Exception:
            return None

        return None

    def _detect_template_contour(self, image):
        """Return the dominant white template contour, or None if unavailable."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        _, white_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        white_mask = cv2.dilate(white_mask, kernel, iterations=3)
        white_mask = cv2.erode(white_mask, kernel, iterations=1)

        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        img_area = image.shape[0] * image.shape[1]
        if area < img_area * 0.20:
            return None

        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(max(w, h)) / float(min(w, h))
        if aspect_ratio < 1.1 or aspect_ratio > 1.7:
            return None

        return largest_contour

    @staticmethod
    def _order_points(points):
        """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)
        point_sums = points.sum(axis=1)
        point_diffs = np.diff(points, axis=1)

        rect[0] = points[np.argmin(point_sums)]
        rect[2] = points[np.argmax(point_sums)]
        rect[1] = points[np.argmin(point_diffs)]
        rect[3] = points[np.argmax(point_diffs)]
        return rect

    def _detect_a5_template(self, image):
        """
        Detect the white calibration template/background in the image.
        The template is approximately 113.52mm × 180.41mm in the current setup.
        
        Returns:
            tuple: (x1, y1, x2, y2) of detected template, or None if not found
        """
        largest_contour = self._detect_template_contour(image)
        if largest_contour is None:
            return None

        # Get bounding rectangle of the template
        x, y, w, h = cv2.boundingRect(largest_contour)

        return (x, y, x + w, y + h)

    def _calibrate_from_a5_template(self, image):
        """
        Calibrate pixel-to-mm ratio using the white template.
        Template dimensions: 113.52mm (width) × 180.41mm (height)
        
        Returns:
            float: pixels_per_mm calibration factor, or None if template not found
        """
        template_box = self._detect_a5_template(image)
        if template_box is None:
            return None
        
        x1, y1, x2, y2 = template_box
        template_width_px = max(1, x2 - x1)
        template_height_px = max(1, y2 - y1)

        # Template dimensions in mm
        template_width_mm = self.TEMPLATE_WIDTH_MM
        template_height_mm = self.TEMPLATE_HEIGHT_MM

        # Determine if template is in portrait (height > width) or landscape.
        if template_height_px > template_width_px:
            # Portrait orientation: height = 180.41mm, width = 113.52mm
            pixels_per_mm_y = template_height_px / template_height_mm
            pixels_per_mm_x = template_width_px / template_width_mm
        else:
            # Landscape orientation: height = 113.52mm, width = 180.41mm
            pixels_per_mm_y = template_height_px / template_width_mm
            pixels_per_mm_x = template_width_px / template_height_mm

        # Use average of x/y for backward compatibility, but keep the axes.
        pixels_per_mm = (pixels_per_mm_x + pixels_per_mm_y) / 2.0

        print(
            f"   [SIZE] White template detected: {template_width_px}x{template_height_px}px, "
            f"pixels_per_mm={pixels_per_mm:.2f} (x={pixels_per_mm_x:.2f}, y={pixels_per_mm_y:.2f})"
        )

        return {
            'x': pixels_per_mm_x,
            'y': pixels_per_mm_y,
            'avg': pixels_per_mm,
        }

    def _crop_to_white_template(self, image):
        """Crop the input image to the detected white calibration plate."""
        contour = self._detect_template_contour(image)
        if contour is None:
            return image, {
                'applied': False,
                'box': None,
                'original_shape': {'height': image.shape[0], 'width': image.shape[1]},
                'cropped_shape': {'height': image.shape[0], 'width': image.shape[1]},
            }

        x, y, w, h = cv2.boundingRect(contour)
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image.shape[1], x + w)
        y2 = min(image.shape[0], y + h)

        if x2 <= x1 or y2 <= y1:
            return image, {
                'applied': False,
                'box': None,
                'original_shape': {'height': image.shape[0], 'width': image.shape[1]},
                'cropped_shape': {'height': image.shape[0], 'width': image.shape[1]},
            }

        cropped = image[y1:y2, x1:x2].copy()
        return cropped, {
            'applied': True,
            'box': [int(x1), int(y1), int(x2), int(y2)],
            'original_shape': {'height': image.shape[0], 'width': image.shape[1]},
            'cropped_shape': {'height': cropped.shape[0], 'width': cropped.shape[1]},
        }

    def _calibrate_from_cropped_plate(self, image, crop_info=None):
        """
        Calibrate using the fixed white plate dimensions after cropping.
        Plate size is 180.41mm x 113.52mm; orientation is inferred from pixels.
        """
        if crop_info and not crop_info.get('applied'):
            return None

        plate_width_px = max(1, image.shape[1])
        plate_height_px = max(1, image.shape[0])

        if plate_height_px > plate_width_px:
            pixels_per_mm_y = plate_height_px / self.TEMPLATE_HEIGHT_MM
            pixels_per_mm_x = plate_width_px / self.TEMPLATE_WIDTH_MM
            plate_width_mm = self.TEMPLATE_WIDTH_MM
            plate_height_mm = self.TEMPLATE_HEIGHT_MM
        else:
            pixels_per_mm_y = plate_height_px / self.TEMPLATE_WIDTH_MM
            pixels_per_mm_x = plate_width_px / self.TEMPLATE_HEIGHT_MM
            plate_width_mm = self.TEMPLATE_HEIGHT_MM
            plate_height_mm = self.TEMPLATE_WIDTH_MM

        pixels_per_mm = (pixels_per_mm_x + pixels_per_mm_y) / 2.0
        
        # Compare measured plate aspect ratio (pixels) with expected physical aspect ratio
        cropped_ratio = max(plate_width_px, plate_height_px) / min(plate_width_px, plate_height_px)
        expected_ratio = self.TEMPLATE_HEIGHT_MM / self.TEMPLATE_WIDTH_MM
        ratio_diff_percent = abs(cropped_ratio - expected_ratio) / expected_ratio * 100
        print(
            f"   [CALIBRATION] Aspect Ratio Comparison: Cropped Plate = {cropped_ratio:.3f}, "
            f"Expected = {expected_ratio:.3f} (Diff: {ratio_diff_percent:.1f}%)"
        )
        
        print(
            f"   [SIZE] Cropped white plate: {plate_width_px}x{plate_height_px}px, "
            f"plate={plate_width_mm:.2f}x{plate_height_mm:.2f}mm, "
            f"pixels_per_mm={pixels_per_mm:.2f} (x={pixels_per_mm_x:.2f}, y={pixels_per_mm_y:.2f})"
        )

        return {
            'x': pixels_per_mm_x,
            'y': pixels_per_mm_y,
            'avg': pixels_per_mm,
            'plate_width_mm': plate_width_mm,
            'plate_height_mm': plate_height_mm,
            'plate_width_px': plate_width_px,
            'plate_height_px': plate_height_px,
        }

    @staticmethod
    def _direction_angle_from_contour(contour):
        """
        Estimate the major-axis direction angle in degrees for a contour.
        Returns an angle in image coordinates, measured counterclockwise from +x.
        """
        pts = contour.reshape(-1, 2).astype(np.float32)
        if len(pts) < 5:
            return 0.0

        mean, eigenvectors = cv2.PCACompute(pts, mean=None, maxComponents=2)
        if eigenvectors is None or len(eigenvectors) == 0:
            return 0.0

        vx, vy = eigenvectors[0]
        angle = float(np.degrees(np.arctan2(vy, vx)))
        return angle

    def _color_name_from_hsv(self, h, s, v):
        """Map average HSV values to a readable color name (optimized for coffee beans)."""
        # Very dark -> Black
        if v < 40:
            return "Dark Brown"
        
        # Nearly grayscale (low saturation) -> Gray/White
        if s < 15:
            if v > 180:
                return "Light Gray"
            if v > 120:
                return "Gray"
            return "Dark Gray"
        
        # Coffee beans are primarily in brown/yellow hue range (OpenCV: 0-180 scale)
        # Red-Brown hues (0-5 or 175-180)
        if h <= 5 or h >= 175:
            return "Dark Brown"
        
        # Orange-Brown (5-15): darker browns with warm tone
        if h < 15:
            if v < 80:
                return "Dark Brown"
            return "Brown"
        
        # Brown (15-30): classic brown coffee beans
        if h < 30:
            if v < 70:
                return "Dark Brown"
            if s > 100:
                return "Brown"
            return "Light Brown"
        
        # Yellow-Brown (30-40): lighter brown with yellow tone
        if h < 40:
            return "Light Brown"
        
        # Yellow tones (40-60): rare but possible in lighter roasts
        if h < 60:
            return "Yellow"
        
        # Green/Blue tones (60+): typically not coffee, but keep for non-beans
        if h < 85:
            return "Green"
        if h < 130:
            return "Blue"
        
        return "Red"

    def _extract_color_info(self, image, box):
        """Extract mean color statistics from a detection box."""
        x1, y1, x2, y2 = map(int, box)
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return {
                'color_name': 'Unknown',
                'rgb': [0, 0, 0],
                'hex': '#000000',
                'hsv': {'h': 0.0, 's': 0.0, 'v': 0.0},
                'gray_std': 0.0,
                'warm_score': 0.0,
                'edge_ratio': 0.0,
            }

        roi = image[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 120)

        bgr = np.mean(roi, axis=(0, 1))
        rgb = [int(round(float(bgr[2]))), int(round(float(bgr[1]))), int(round(float(bgr[0])))]
        h_mean = float(np.mean(hsv[:, :, 0]))
        s_mean = float(np.mean(hsv[:, :, 1]))
        v_mean = float(np.mean(hsv[:, :, 2]))
        gray_std = float(np.std(gray))
        warm_score = float(np.mean(lab[:, :, 1].astype(np.float32) + lab[:, :, 2].astype(np.float32)))
        edge_ratio = float(np.mean(edges > 0))

        # Get initial color from HSV
        color_name = self._color_name_from_hsv(h_mean, s_mean, v_mean)
        
        # LAB color space correction: if warmth is high and saturation is moderate, likely brown coffee bean
        # This corrects for HSV hue misclassification in certain lighting conditions
        if warm_score > 305 and 50 < v_mean < 180 and s_mean > 25:
            if color_name in ['Green', 'Blue', 'Red']:
                color_name = 'Brown' if v_mean > 80 else 'Dark Brown'

        return {
            'color_name': color_name,
            'rgb': rgb,
            'hex': f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}",
            'hsv': {'h': h_mean, 's': s_mean, 'v': v_mean},
            'gray_std': gray_std,
            'warm_score': warm_score,
            'edge_ratio': edge_ratio,
        }

    def _classify_object_type(self, box, color_info):
        """
        Classify a detected object as coffee_bean or non_bean using color/texture heuristics.
        """
        x1, y1, x2, y2 = map(int, box)
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        aspect = float(bw) / float(bh)

        s_mean = float(color_info['hsv']['s'])
        v_mean = float(color_info['hsv']['v'])
        warm_score = float(color_info['warm_score'])
        gray_std = float(color_info['gray_std'])

        # Rice/neutral grains and very pale smooth objects fall here.
        if s_mean < 20 and warm_score < 262 and v_mean > 115 and gray_std < 45:
            return 'non_bean'
        if s_mean < 15 and warm_score < 266 and v_mean > 105:
            return 'non_bean'
        if s_mean < 24 and warm_score < 264 and (aspect > 1.6 or aspect < 0.62):
            return 'non_bean'

        return 'coffee_bean'

    def _estimate_size_mm(self, image, box, pixels_per_mm):
        """
        Estimate object dimensions in millimetres.

        Tries to recover the bean silhouette inside the detection box so the
        reported size is based on the object's shape, not just the padded box.
        Falls back to the clipped box dimensions if contour extraction fails.
        """
        ppm_x = None
        ppm_y = None
        if isinstance(pixels_per_mm, dict):
            ppm_x = float(pixels_per_mm.get('x') or pixels_per_mm.get('avg') or 0)
            ppm_y = float(pixels_per_mm.get('y') or pixels_per_mm.get('avg') or 0)
            ppm_avg = float(pixels_per_mm.get('avg') or 0)
        else:
            ppm_avg = float(pixels_per_mm or 0)
            ppm_x = ppm_avg
            ppm_y = ppm_avg

        if ppm_avg <= 0:
            return None

        x1, y1, x2, y2 = map(int, box)
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            short_mm = float(round(min(bw / max(ppm_x, 1e-6), bh / max(ppm_y, 1e-6)), 1))
            long_mm = float(round(max(bw / max(ppm_x, 1e-6), bh / max(ppm_y, 1e-6)), 1))
            return {'width': short_mm, 'height': long_mm}

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        masks = []
        try:
            _, bean_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            masks.append(bean_mask)
        except Exception:
            pass

        try:
            adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
            )
            masks.append(adaptive)
        except Exception:
            pass

        try:
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            warmth = lab[:, :, 1].astype(np.int16) + lab[:, :, 2].astype(np.int16)
            warm_threshold = np.percentile(warmth, 60)
            warm_mask = (warmth > warm_threshold).astype(np.uint8) * 255
            masks.append(warm_mask)
        except Exception:
            pass

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        best_contour = None
        best_area = 0.0
        roi_area = float(max(1, bw * bh))

        for mask in masks:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area < 10.0:
                    continue
                area_ratio = area / roi_area
                if area_ratio < 0.03 or area_ratio > 0.98:
                    continue
                if area > best_area:
                    best_area = area
                    best_contour = cnt

        if best_contour is not None:
            rect = cv2.minAreaRect(best_contour)
            rect_w, rect_h = rect[1]
            if rect_w > 1 and rect_h > 1:
                major_px = max(rect_w, rect_h)
                minor_px = min(rect_w, rect_h)
                major_angle = self._direction_angle_from_contour(best_contour)
                minor_angle = major_angle + 90.0

                def _axis_mm(length_px, angle_deg):
                    angle_rad = np.radians(angle_deg)
                    effective = np.sqrt(
                        (np.cos(angle_rad) / max(ppm_x, 1e-6)) ** 2 +
                        (np.sin(angle_rad) / max(ppm_y, 1e-6)) ** 2
                    )
                    return length_px * effective

                major_mm = _axis_mm(major_px, major_angle)
                minor_mm = _axis_mm(minor_px, minor_angle)
                short_mm = float(round(min(minor_mm, major_mm), 1))
                long_mm = float(round(max(minor_mm, major_mm), 1))
                return {'width': short_mm, 'height': long_mm}

        short_mm = float(round(min(bw / max(ppm_x, 1e-6), bh / max(ppm_y, 1e-6)), 1))
        long_mm = float(round(max(bw / max(ppm_x, 1e-6), bh / max(ppm_y, 1e-6)), 1))
        return {'width': short_mm, 'height': long_mm}

    def _extract_polygon_from_box(self, image, box, debug=False):
        """
        Given an image and a bounding box [x1,y1,x2,y2], extract the largest contour
        within the box and return it as a polygon (list of (x,y) points in image coords).
        Falls back to rectangular polygon if no contour is found.
        """
        x1, y1, x2, y2 = box
        h, w = image.shape[:2]
        # Clip box
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        crop = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # CLAHE to improve local contrast
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
        except Exception:
            pass

        # Try multiple thresholding strategies and pick best contours
        strategies = []
        # Otsu (smoothed)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        strategies.append(th_otsu)
        # Adaptive mean
        th_adapt_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY, 31, 5)
        strategies.append(th_adapt_mean)
        # Adaptive gaussian
        th_adapt_gauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 31, 5)
        strategies.append(th_adapt_gauss)

        # Add a color-based mask (use LAB to bias towards warm bean colors)
        try:
            lab_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            a = lab_crop[:, :, 1].astype(np.int16)
            b = lab_crop[:, :, 2].astype(np.int16)
            warm_score = (a + b).astype(np.int16)
            warm_t = np.percentile(warm_score, 65)
            warm_mask = (warm_score > warm_t).astype(np.uint8) * 255
            strategies.append(warm_mask)
        except Exception:
            pass

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        best_contours = []
        box_area = max(1, (x2 - x1) * (y2 - y1))

        for th in strategies:
            th2 = th.copy() if th.dtype == np.uint8 else th.astype(np.uint8)
            th2 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel, iterations=1)
            th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel, iterations=1)
            contours, _ = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # filter contours by area ratio and solidity
            filtered = []
            for cnt in contours:
                ca = cv2.contourArea(cnt)
                if ca <= 0:
                    continue
                area_ratio = ca / float(box_area)
                # compute solidity
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull) if len(hull) > 2 else ca
                solidity = ca / hull_area if hull_area > 0 else 0
                # accept contours that are neither too tiny nor nearly the whole box
                if area_ratio < 0.005 or area_ratio > 0.98:
                    continue
                if solidity < 0.20:
                    continue
                filtered.append((cnt, ca, area_ratio, solidity))

            if filtered:
                # sort by area (desc) and keep multiple candidate contours
                filtered.sort(key=lambda x: (-x[1], abs(x[2] - 0.05)))
                best_contours = filtered
                break

        # If we still have no contours, try Canny edges as a last-ditch effort
        if not best_contours:
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.dilate(edges, kernel, iterations=1)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered = []
            for cnt in contours:
                ca = cv2.contourArea(cnt)
                if ca <= 0:
                    continue
                area_ratio = ca / float(box_area)
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull) if len(hull) > 2 else ca
                solidity = ca / hull_area if hull_area > 0 else 0
                if area_ratio < 0.004 or area_ratio > 0.98:
                    continue
                if solidity < 0.18:
                    continue
                filtered.append((cnt, ca, area_ratio, solidity))
            if filtered:
                filtered.sort(key=lambda x: (-x[1], abs(x[2] - 0.05)))
                best_contours = filtered

        # If still nothing, try watershed on the crop to split regions
        if not best_contours:
            try:
                # prepare mask for watershed using Otsu on blurred gray
                blur2 = cv2.GaussianBlur(gray, (7, 7), 0)
                _, th_ws = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                th_ws = cv2.morphologyEx(th_ws, cv2.MORPH_OPEN, kernel, iterations=1)
                dist = cv2.distanceTransform(th_ws, cv2.DIST_L2, 5)
                _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
                sure_fg = np.uint8(sure_fg)
                unknown = cv2.subtract(th_ws, sure_fg)
                _, markers = cv2.connectedComponents(sure_fg)
                markers = markers + 1
                markers[unknown == 255] = 0
                color_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                markers = cv2.watershed(color_crop, markers)
                # extract contours from each marker region
                polys_from_ws = []
                for m in np.unique(markers):
                    if m <= 1:
                        continue
                    mask = np.uint8(markers == m) * 255
                    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in cnts:
                        ca = cv2.contourArea(cnt)
                        if ca <= 2:
                            continue
                        hull = cv2.convexHull(cnt)
                        hull_area = cv2.contourArea(hull) if len(hull) > 2 else ca
                        solidity = ca / hull_area if hull_area > 0 else 0
                        area_ratio = ca / float(box_area)
                        if area_ratio < 0.003 or solidity < 0.18:
                            continue
                        polys_from_ws.append((cnt, ca, area_ratio, solidity))
                if polys_from_ws:
                    polys_from_ws.sort(key=lambda x: -x[1])
                    best_contours = polys_from_ws
            except Exception:
                pass

        if not best_contours:
            # fallback rectangle if nothing suitable found
            return [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]

        # Build polygons for up to N contours (split box into multiple beans if present)
        polys = []
        max_polys = 8
        for idx, (cnt, ca, area_ratio, solidity) in enumerate(best_contours[:max_polys]):
            peri = cv2.arcLength(cnt, True)
            # use a smaller epsilon for tight shapes but allow small minimum
            eps = max(1.0, 0.005 * peri)
            approx = cv2.approxPolyDP(cnt, eps, True)
            # if approx has too few points, fallback to raw contour sampling
            pts = approx.reshape(-1, 2) if approx is not None and len(approx) >= 3 else cnt.reshape(-1, 2)
            poly = []
            for p in pts:
                px, py = int(p[0]) + x1, int(p[1]) + y1
                # clip
                px = max(0, min(px, w - 1))
                py = max(0, min(py, h - 1))
                poly.append([px, py])
            # ensure polygon has area and at least 3 points
            if len(poly) >= 3 and cv2.contourArea(np.array(poly)) > 1.0:
                polys.append(poly)

        if debug:
            print(f"Extracted {len(polys)} polygon(s) from box {box}, first_area_ratio={best_contours[0][2]:.4f}, solidity={best_contours[0][3]:.3f}")

        return polys

    def _detect_beans_by_contours(self, image, debug=False):
        """
        Detect individual beans using scale-aware morphology and color filtering.
        This fallback is used when model detections are coarse/low-confidence.
        """
        h, w = image.shape[:2]
        img_area = float(h * w)

        # Warm-color mask (coffee beans are generally warmer than paper/cloth background).
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        warm = lab[:, :, 1].astype(np.int16) + lab[:, :, 2].astype(np.int16)
        warm_threshold = float(np.percentile(warm, 70))
        warm_mask = (warm > warm_threshold).astype(np.uint8) * 255

        # Local contrast mask to highlight dark bean bodies against nearby background.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Scale kernel with image size so zoomed-in and full-frame shots both work.
        kernel_size = int(max(31, min(81, ((min(h, w) * 0.09) // 2) * 2 + 1)))
        local_bg = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        dark_delta = cv2.subtract(local_bg, gray)
        _, dark_mask = cv2.threshold(dark_delta, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        bean_mask = cv2.bitwise_and(dark_mask, warm_mask)

        k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bean_mask = cv2.morphologyEx(bean_mask, cv2.MORPH_OPEN, k_clean, iterations=1)
        bean_mask = cv2.morphologyEx(bean_mask, cv2.MORPH_CLOSE, k_clean, iterations=1)
        
        # Use watershed to separate touching/merged beans
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            sure_bg = cv2.dilate(bean_mask, kernel, iterations=2)
            dist_transform = cv2.distanceTransform(bean_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            markers = cv2.watershed(cv2.cvtColor(bean_mask, cv2.COLOR_GRAY2BGR), markers)
        except Exception:
            markers = None

        contours, _ = cv2.findContours(bean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        # First pass: gather plausible bean-like contours.
        max_area = img_area * 0.03
        candidates = []
        largest_area = 0.0
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < 10.0 or area > max_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw <= 1 or bh <= 1:
                continue

            aspect = float(bw) / float(bh)
            if aspect < 0.2 or aspect > 5.0:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull))
            solidity = area / hull_area if hull_area > 0 else 0.0
            # Higher solidity threshold to filter out noise/texture
            if solidity < 0.50:
                continue

            largest_area = max(largest_area, area)
            candidates.append((cnt, area, solidity))

        if not candidates:
            return []

        # Dynamic minimum area: stricter to avoid picking up texture/shadows
        min_area = max(50.0, img_area * 0.00003, largest_area * 0.08)

        contour_detections = []
        for cnt, area, solidity in candidates:
            if area < min_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)

            aspect = float(bw) / float(bh)
            if aspect < 0.2 or aspect > 5.0:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
            # Higher circularity threshold to filter thin/noise patterns
            if circularity < 0.25:
                continue

            eps = max(1.0, 0.02 * perimeter)
            approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
            poly = [[int(px), int(py)] for px, py in approx]
            if len(poly) < 3:
                poly = [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]

            # If contour mostly traces an inner crease, expand to a safer outer box.
            box_area = float(max(1, bw * bh))
            fill_ratio = area / box_area
            if fill_ratio < 0.45:
                pad_x = max(2, int(bw * 0.35))
                pad_y = max(2, int(bh * 0.35))
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w - 1, x + bw + pad_x)
                y2 = min(h - 1, y + bh + pad_y)
                x = x1
                y = y1
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

            confidence = round(min(0.99, 0.45 + (0.35 * solidity) + (0.2 * min(1.0, circularity))), 3)
            contour_detections.append({
                'class': 'coffee_bean',
                'confidence': confidence,
                'box': [int(x), int(y), int(x + bw), int(y + bh)],
                'polygon': poly,
            })

        # Keep one detection per bean-like blob.
        contour_detections = self._apply_nms(contour_detections, iou_threshold=0.3)
        contour_detections.sort(key=lambda d: (d['box'][1], d['box'][0]))

        if debug:
            print(
                f"Contour fallback: candidates={len(contour_detections)}, "
                f"min_area={min_area:.1f}, max_area={max_area:.1f}, kernel={kernel_size}, "
                f"warm_t={warm_threshold:.1f}"
            )

        return contour_detections

    def _should_use_contour_fallback(self, model_detections, contour_detections, image_shape, debug=False):
        """Decide whether contour detections are more reliable than model detections."""
        if not contour_detections:
            return False

        model_count = len(model_detections)
        contour_count = len(contour_detections)

        # Guard against pathological contour explosion.
        if contour_count > 250:
            return False

        if model_count == 0:
            return True

        h, w = image_shape[:2]
        img_area = float(h * w) if h > 0 and w > 0 else 1.0
        avg_conf = float(np.mean([d.get('confidence', 0.0) for d in model_detections]))
        large_boxes = 0
        for d in model_detections:
            x1, y1, x2, y2 = d['box']
            area_ratio = ((x2 - x1) * (y2 - y1)) / img_area
            if area_ratio > 0.02:
                large_boxes += 1
        large_box_ratio = large_boxes / float(model_count) if model_count > 0 else 0.0

        # The custom bean model often emits low-confidence oversized boxes on hard images.
        poor_model = avg_conf < 0.12 or large_box_ratio > 0.30
        if poor_model:
            use_fallback = contour_count > 0
        else:
            ratio = contour_count / float(model_count) if model_count > 0 else 0.0
            use_fallback = contour_count >= 3 and 0.6 <= ratio <= 1.8

        if debug:
            print(
                f"Fallback decision: model_count={model_count}, contour_count={contour_count}, "
                f"avg_conf={avg_conf:.3f}, large_box_ratio={large_box_ratio:.2f}, use={use_fallback}"
            )

        return use_fallback

    def _segment_with_sam2(self, image, boxes):
        """
        Run SAM 2 to generate high-fidelity binary masks for each detected
        bounding box using **centroid point prompts** and chunked batched
        inference.

        Centroid-Prompting Strategy
        ──────────────────────────
        For each YOLO bounding box the centroid is calculated:
            cx = (x1 + x2) / 2,  cy = (y1 + y2) / 2
        Both the centroid (as a foreground point, label=1) and the original
        bounding box are fed to SAM 2 together.  The box constrains the
        spatial region so SAM 2 does not leak into the background, while
        the centroid guides the model to the object's centre for a tighter,
        more accurate instance mask.

        Optimization
        ────────────
        • The image is embedded **once** (the most expensive step).
        • Centroid prompts are grouped into chunks of at most
          ``SAM_CHUNK_SIZE`` (default 64) and each chunk runs through the
          SAM 2 decoder in a single forward pass, keeping peak VRAM usage
          flat regardless of how many beans are in the image.
        • All tensor operations run inside ``torch.autocast`` with FP16.
        • PyTorch SDPA (Flash-Attention / memory-efficient attention) is
          pre-enabled at init time to further reduce attention memory.

        Args:
            image (np.ndarray): Full image in BGR format (H × W × 3).
            boxes (list[list[int]]): List of [x1, y1, x2, y2] bounding boxes.

        Returns:
            list[np.ndarray | None]: Binary uint8 masks (255 = bean, 0 = bg),
                one per box.  Returns None for a box when extraction fails.
        """
        if self.sam2_predictor is None or len(boxes) == 0:
            return [None] * len(boxes)

        try:
            # SAM 2 expects RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

            with torch.autocast(device_type=device_type, dtype=torch.float16):
                # ── Embed image once (encoder forward pass) ──
                self.sam2_predictor.set_image(image_rgb)

                # ── Chunked decoder passes ──
                result = []
                chunk_size = self.SAM_CHUNK_SIZE
                for chunk_start in range(0, len(boxes), chunk_size):
                    chunk_boxes = boxes[chunk_start: chunk_start + chunk_size]

                    # Calculate centroids for each box in this chunk
                    # point_coords shape: (B, 1, 2) — one point per object
                    # point_labels shape: (B, 1)    — 1 = foreground
                    centroids = np.array([
                        [[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]]
                        for b in chunk_boxes
                    ], dtype=np.float32)  # (B, 1, 2)
                    fg_labels = np.ones(
                        (len(chunk_boxes), 1), dtype=np.int32
                    )  # (B, 1)
                    boxes_arr = np.array(chunk_boxes, dtype=np.float32)

                    mask_input, unnorm_coords, labels, unnorm_box = (
                        self.sam2_predictor._prep_prompts(
                            point_coords=centroids,
                            point_labels=fg_labels,
                            box=boxes_arr,
                            mask_logits=None,
                            normalize_coords=True,
                        )
                    )

                    # Single decoder pass for this chunk → (C, 1, H, W)
                    masks_t, _iou_t, _ = self.sam2_predictor._predict(
                        unnorm_coords,
                        labels,
                        unnorm_box,
                        mask_input,
                        multimask_output=False,
                        return_logits=False,
                    )

                    # (C, H, W) bool → numpy, then crop to each box ROI
                    masks_np = masks_t.squeeze(1).cpu().numpy()
                    for mask_bool, box in zip(masks_np, chunk_boxes):
                        x1, y1, x2, y2 = [int(v) for v in box]
                        roi_mask = mask_bool[y1:y2, x1:x2].astype(np.uint8) * 255
                        result.append(roi_mask)

            n_valid = sum(m is not None and m.size > 0 for m in result)
            n_chunks = (len(boxes) + chunk_size - 1) // chunk_size
            print(f'   [SAM2] {n_valid}/{len(boxes)} masks via {n_chunks} chunk(s) '
                  f'(chunk_size={chunk_size}, prompt=centroid)')
            return result

        except Exception as e:
            print(f'[WARN] SAM 2 segmentation failed: {e}')
            return [None] * len(boxes)

    def _estimate_pca_metrics_from_mask(self, mask, pixels_per_mm):
        """
        Estimate bean dimensions from a high-fidelity binary mask using
        PCA-based orientation alignment and pixel-to-mm conversion.

        Algorithm
        ─────────
        1. Extract the largest contour from the mask.
        2. Run PCA on the contour points to determine the centroid and
           the principal-component (major-axis) direction.
        3. Pad the mask, then rotate it so the major axis is horizontal.
        4. Compute the axis-aligned bounding rectangle of the aligned
           contour → width_px (minor axis) and height_px (major axis).
        5. Divide pixel measurements by the calibrated pixels_per_mm
           to produce real-world millimetre values.

        Args:
            mask (np.ndarray): Binary uint8 ROI mask (255 = bean, 0 = bg).
            pixels_per_mm (float | dict): Calibration factor.  When a dict
                it must contain at least one of ``'x'``, ``'y'``, ``'avg'``.

        Returns:
            dict | None: On success returns::

                {
                    'width_mm':  float,   # minor axis in mm
                    'height_mm': float,   # major axis in mm
                    'width_px':  int,     # minor axis in pixels
                    'height_px': int,     # major axis in pixels
                    'angle_deg': float,   # PCA orientation angle
                    'centroid':  (float, float),
                }

                Returns ``None`` if analysis fails.
        """
        if mask is None or mask.size == 0:
            return None

        # ── Resolve pixels-per-mm ──────────────────────────────────
        if isinstance(pixels_per_mm, dict):
            ppm_x = float(pixels_per_mm.get('x') or pixels_per_mm.get('avg') or 0)
            ppm_y = float(pixels_per_mm.get('y') or pixels_per_mm.get('avg') or 0)
        else:
            ppm_x = float(pixels_per_mm or 0)
            ppm_y = ppm_x
        ppm_avg = (ppm_x + ppm_y) / 2.0
        if ppm_avg <= 0:
            return None

        try:
            # ── 0. Clean up mask edges (centroid prompts can include
            #       a thin halo of shadow / background pixels) ──────
            k_sz = max(3, int(min(mask.shape[:2]) * 0.02) | 1)  # odd, scales with mask
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
            mask_clean = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, kern, iterations=1
            )
            # Extra erosion to peel off the outermost fringe pixels
            kern_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_clean = cv2.erode(mask_clean, kern_erode, iterations=1)

            # ── 1. Largest contour ─────────────────────────────────
            contours, _ = cv2.findContours(
                mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                return None
            cnt = max(contours, key=cv2.contourArea)
            pts = cnt.squeeze(1).astype(np.float32)
            if len(pts) < 5:
                return None

            # ── 2. PCA → centroid & orientation ────────────────────
            mean, eigenvectors = cv2.PCACompute(
                pts, mean=None, maxComponents=2
            )
            if eigenvectors is None or len(eigenvectors) < 1:
                return None
            cx, cy = float(mean[0][0]), float(mean[0][1])
            v1 = eigenvectors[0]
            angle_deg = float(np.degrees(np.arctan2(v1[1], v1[0])))

            # ── 3. Pad + rotate mask so major axis is horizontal ──
            h, w = mask_clean.shape[:2]
            pad = max(h, w)
            padded = cv2.copyMakeBorder(
                mask_clean, pad, pad, pad, pad,
                cv2.BORDER_CONSTANT, value=0,
            )
            cx_p, cy_p = cx + pad, cy + pad
            rot_mat = cv2.getRotationMatrix2D(
                (cx_p, cy_p), angle_deg, 1.0
            )
            aligned = cv2.warpAffine(
                padded, rot_mat,
                (padded.shape[1], padded.shape[0]),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            # ── 4. Bounding rect on aligned mask ──────────────────
            a_contours, _ = cv2.findContours(
                aligned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not a_contours:
                return None
            a_cnt = max(a_contours, key=cv2.contourArea)
            _, _, bw, bh = cv2.boundingRect(a_cnt)
            if bw <= 0 or bh <= 0:
                return None

            short_px = min(bw, bh)
            long_px = max(bw, bh)

            # ── 5. Convert to mm using directional calibration ────
            angle_rad = np.radians(angle_deg)
            ppm_major = np.sqrt(
                (ppm_x * np.cos(angle_rad)) ** 2 +
                (ppm_y * np.sin(angle_rad)) ** 2
            )
            ppm_minor = np.sqrt(
                (ppm_x * np.sin(angle_rad)) ** 2 +
                (ppm_y * np.cos(angle_rad)) ** 2
            )
            if ppm_major <= 0:
                ppm_major = 1.0
            if ppm_minor <= 0:
                ppm_minor = 1.0

            width_mm = float(round(short_px / ppm_minor, 1))
            height_mm = float(round(long_px / ppm_major, 1))

            return {
                'width_mm': width_mm,
                'height_mm': height_mm,
                'width_px': int(short_px),
                'height_px': int(long_px),
                'angle_deg': round(angle_deg, 2),
                'centroid': (cx, cy),
            }
        except Exception:
            return None

    def _estimate_length_mm_from_mask(self, mask, pixels_per_mm):
        """
        Estimate bean length (major axis) in mm from a high-fidelity binary
        mask via PCA-based rotation alignment.

        Args:
            mask (np.ndarray): Binary uint8 ROI mask (255 = bean).
            pixels_per_mm (float | dict): Calibration factor.

        Returns:
            float | None: Length in mm, or None if analysis fails.
        """
        metrics = self._estimate_pca_metrics_from_mask(mask, pixels_per_mm)
        if metrics is None:
            return None
        return metrics['height_mm']  # major axis = length

    def _estimate_size_mm_from_mask(self, mask, pixels_per_mm):
        """
        Estimate bean width and height in mm from a high-fidelity binary mask
        via PCA-based rotation alignment.

        Args:
            mask (np.ndarray): Binary uint8 ROI mask (255 = bean).
            pixels_per_mm (float | dict): Calibration factor.

        Returns:
            dict | None: {'width': float, 'height': float} in mm, or None.
        """
        metrics = self._estimate_pca_metrics_from_mask(mask, pixels_per_mm)
        if metrics is None:
            return None
        return {'width': metrics['width_mm'], 'height': metrics['height_mm']}

    def detect_objects(self, image_path, confidence_threshold=0.25, save_output=None,
                       iou=0.3,
                       min_confidence_output=0.25,
                       min_box_area_ratio=0.0005,
                       max_box_area_ratio=0.7,
                       min_aspect=0.25,
                       max_aspect=4.0,
                       box_shrink_ratio=0.15,
                       use_contour_fallback=False,
                       manual_crop=False,
                       use_sam2=None,
                       debug=False):
        """
        Detect objects in an image and count them by class with extra post-processing.
        
        **SIZE CALIBRATION (WHITE TEMPLATE)**
        For accurate bean size measurements, ensure the image includes the white calibration template/background.
        The template (113.52mm × 180.41mm) is used to automatically calibrate pixel-to-mm conversions.

        Args:
            image_path (str): Path to the input image.
            confidence_threshold (float): Model-level confidence used during inference.
            save_output (str): Path to save annotated image.
            iou (float): IoU used when calling the model and for final NMS.
            min_confidence_output (float): Minimum confidence for final outputs.
            min_box_area_ratio (float): Minimum box area relative to image area to keep.
            max_box_area_ratio (float): Maximum box area relative to image area to keep.
            min_aspect (float): Minimum width/height ratio to keep.
            max_aspect (float): Maximum width/height ratio to keep.
            box_shrink_ratio (float): Shrink boxes by this ratio (0.15 = 15% from each side).
            use_contour_fallback (bool): If True, use contour segmentation when model quality is poor.
            manual_crop (bool): If True, skip auto-cropping and calibrate using full image dimensions.
            use_sam2 (bool | None): Override instance-level use_sam2 flag. None uses the instance default.
            debug (bool): If True, print per-detection filtering decisions.

        Returns:
            dict: Detection results with filtered counts, boxes, and calibrated size measurements.
                  'pixels_per_mm': Calibration factor derived from the template.
                  'size_mm': Each detection includes {'width': mm, 'height': mm} when calibrated.
                  'sam2_active': Whether SAM 2 mask generation was used for this call.
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return {
                'total_count': 0,
                'by_class': {},
                'detections': [],
                'image_path': str(image_path),
                'detection_source': 'none',
            }

        original_image_shape = {'height': image.shape[0], 'width': image.shape[1]}
        if manual_crop:
            crop_info = {
                'applied': True,
                'box': [0, 0, image.shape[1], image.shape[0]],
                'original_shape': {'height': image.shape[0], 'width': image.shape[1]},
                'cropped_shape': {'height': image.shape[0], 'width': image.shape[1]},
            }
        else:
            image, crop_info = self._crop_to_white_template(image)

        h, w = image.shape[:2]
        img_area = float(w * h)
        if debug:
            print(
                f"Image dims from cv2 after plate crop: h={h}, w={w}, "
                f"img_area={img_area}, crop_applied={crop_info.get('applied')}"
            )

        # Run model inference.
        results = self.model(image, conf=confidence_threshold, iou=iou)
        result = results[0]

        # Collect model detections and normalize coordinates.
        raw_detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            try:
                xyxy_arr = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy().astype(int)
            except Exception:
                xyxy_arr = []
                confs = []
                clss = []
                for box in result.boxes:
                    try:
                        xyxy_arr.append(box.xyxy[0].cpu().numpy())
                        confs.append(float(box.conf[0]))
                        clss.append(int(box.cls[0]))
                    except Exception:
                        continue

            orig_h, orig_w = h, w
            for coords, conf, cid in zip(xyxy_arr, confs, clss):
                x1, y1, x2, y2 = coords.tolist()
                x1 = max(0, min(x1, orig_w))
                x2 = max(0, min(x2, orig_w))
                y1 = max(0, min(y1, orig_h))
                y2 = max(0, min(y2, orig_h))

                coords_clipped = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
                class_id = int(cid)
                class_name = result.names.get(class_id, str(class_id))

                raw_detections.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'box': coords_clipped,
                })

        # Area/aspect/confidence filtering.
        prefiltered = []
        for d in raw_detections:
            x1, y1, x2, y2 = d['box']
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            area = float(bw * bh)
            area_ratio = area / img_area if img_area > 0 else 0.0
            aspect = float(bw) / float(bh) if bh > 0 else 0.0

            # Temporarily disable strict confidence and tiny-box filtering to avoid
            # missing a small number of beans in edge-case photos. Preserve the
            # original checks commented below for easy re-enable.
            # if d['confidence'] < min_confidence_output:
            #     if debug:
            #         print(f"Discarding by conf {d['confidence']:.4f} < {min_confidence_output}")
            #     continue
            # if area_ratio < min_box_area_ratio:
            #     if debug:
            #         print(f"Discarding tiny box area_ratio={area_ratio:.6f}")
            #     continue
            # Temporarily disable large-box and aspect-ratio filtering to avoid
            # losing beans near tray edges or atypical orientations. Original
            # checks preserved below for re-enable.
            # if area_ratio > max_box_area_ratio:
            #     if debug:
            #         print(f"Discarding large box area_ratio={area_ratio:.3f}")
            #     continue
            # if aspect < min_aspect or aspect > max_aspect:
            #     if debug:
            #         print(f"Discarding by aspect ratio={aspect:.3f}")
            #     continue

            d_out = d.copy()
            d_out['confidence'] = round(d_out['confidence'], 3)
            d_out['box'] = self._shrink_box(d_out['box'], shrink_ratio=box_shrink_ratio)
            prefiltered.append(d_out)

        # Keep track of raw vs filtered boxes for debugging purposes
        # Use a very high IoU threshold temporarily to avoid NMS removing
        # valid nearby beans in dense trays.
        filtered_detections = self._apply_nms(prefiltered, iou_threshold=1.01)
        try:
            raw_boxes = [tuple(map(int, d['box'])) for d in raw_detections]
            pref_boxes = [tuple(map(int, d['box'])) for d in prefiltered]
            filt_boxes = [tuple(map(int, d['box'])) for d in filtered_detections]
            removed_raw_vs_pref = [b for b in raw_boxes if b not in pref_boxes]
            removed_pref_vs_filt = [b for b in pref_boxes if b not in filt_boxes]
            if debug and (removed_raw_vs_pref or removed_pref_vs_filt):
                print(f"   [DEBUG] raw={len(raw_boxes)}, pref={len(pref_boxes)}, filt={len(filt_boxes)}")
                if removed_raw_vs_pref:
                    print(f"   [DEBUG] removed by prefilter: {removed_raw_vs_pref}")
                if removed_pref_vs_filt:
                    print(f"   [DEBUG] removed by NMS: {removed_pref_vs_filt}")
        except Exception:
            pass

        # Keep one detection per model box (no polygon splitting to avoid double counting).
        model_detections = []
        for det in filtered_detections:
            x1, y1, x2, y2 = det['box']
            det_copy = det.copy()
            det_copy['polygon'] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            model_detections.append(det_copy)

        contour_detections = []
        use_fallback = False
        if use_contour_fallback:
            contour_detections = self._detect_beans_by_contours(image, debug=debug)
            use_fallback = self._should_use_contour_fallback(
                model_detections,
                contour_detections,
                image.shape,
                debug=debug,
            )

        final_detections = contour_detections if use_fallback else model_detections
        final_detections.sort(key=lambda d: (d['box'][1], d['box'][0]))
        detection_source = 'contour_fallback' if use_fallback else 'model'

        # --- SAM 2 Mask Generation ---
        _use_sam2 = self.use_sam2 if use_sam2 is None else bool(use_sam2)
        sam2_masks = {}
        if _use_sam2 and self.sam2_predictor is not None and final_detections:
            raw_boxes_for_sam2 = [det['box'] for det in final_detections]
            masks = self._segment_with_sam2(image, raw_boxes_for_sam2)
            sam2_masks = {i: m for i, m in enumerate(masks)}
            print(f'   [SAM2] Generated {sum(m is not None for m in masks)} high-fidelity masks')

        # Enrich detections with object type and color info.
        enriched_detections = []
        bean_count = 0
        non_bean_count = 0
        # Explicitly mark these as NON-BEANS (foreign matter, debris, etc.).
        # Bean-like defect classes stay counted as coffee beans and are tracked
        # separately through defect_type for grading/reporting.
        NON_BEAN_CLASSES = {
            'foreign', 'husk', 'fraghusk', 'debris', 'dust', 'fiber', 'leaf',
            'stick', 'wood', 'stone', 'glass', 'metal', 'plastic', 'contamination',
        }

        color_distribution = {}
        all_boxes = [det['box'] for det in final_detections]
        for idx, det in enumerate(final_detections):
            det_copy = det.copy()
            sam2_mask = sam2_masks.get(idx)
            if sam2_mask is not None:
                det_copy['sam2_mask_data'] = sam2_mask
                contours, _ = cv2.findContours(sam2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    x1, y1, x2, y2 = det_copy['box']
                    cnt_abs = cnt + np.array([x1, y1])
                    det_copy['polygon'] = cnt_abs.squeeze(1).tolist()

            color_info = self._extract_color_info(image, det_copy['box'])
            model_class = det.get('class', '').lower()

            # Use NON_BEAN_CLASSES as a conservative blacklist. If a detector
            # label is suspicious but still bean-like, keep it counted as a bean.
            if model_class in NON_BEAN_CLASSES:
                object_type = self._classify_object_type(det_copy['box'], color_info)
            else:
                # Default to coffee_bean (includes all defect types: black, broken, etc.)
                object_type = 'coffee_bean'

            det_copy['object_type'] = object_type
            det_copy['defect_type'] = model_class  # preserve original model label
            det_copy['class'] = object_type
            det_copy['color'] = {
                'name': color_info['color_name'],
                'hex': color_info['hex'],
                'rgb': color_info['rgb'],
                'hsv': color_info['hsv'],
            }
            enriched_detections.append(det_copy)

            if object_type == 'coffee_bean':
                bean_count += 1
                cname = color_info['color_name']
                color_distribution[cname] = color_distribution.get(cname, 0) + 1
            else:
                non_bean_count += 1

        # Final NMS pass to catch any remaining overlapping boxes. Use a higher
        # IoU threshold to be less aggressive about removing nearby/adjacent
        # beans in tightly packed trays.
        # Temporarily disable final NMS (set threshold >1.0) to avoid dropping
        # any nearby detections in tightly packed trays. This is reversible.
        enriched_detections = self._apply_nms(enriched_detections, iou_threshold=1.01)
        # Recount after final NMS
        bean_count = sum(1 for d in enriched_detections if d.get('object_type') == 'coffee_bean')
        non_bean_count = sum(1 for d in enriched_detections if d.get('object_type') != 'coffee_bean')

        # --- Size estimation: use the cropped white plate as the only calibration reference ---
        pixels_per_mm = None
        pixels_per_mm_x = None
        pixels_per_mm_y = None
        
        calibration = self._calibrate_from_cropped_plate(image, crop_info)

        if isinstance(calibration, dict):
            pixels_per_mm = calibration.get('avg')
            pixels_per_mm_x = calibration.get('x')
            pixels_per_mm_y = calibration.get('y')
        else:
            pixels_per_mm = calibration

        if pixels_per_mm is None or pixels_per_mm == 0.0:
            current_scale = self.TEMPLATE_HEIGHT_MM / 180.41
            fallback_ppm = (max(image.shape[0], image.shape[1]) / 204.0) / current_scale
            pixels_per_mm = fallback_ppm
            pixels_per_mm_x = fallback_ppm
            pixels_per_mm_y = fallback_ppm
            print(f"   [FALLBACK] White plate calibration failed. Using camera resolution fallback pixels_per_mm = {pixels_per_mm:.2f} (scale={current_scale:.4f})")
        
        # Debug: Show what classes were detected
        detected_classes = {}
        for det in enriched_detections:
            defect_type = det.get('defect_type', 'unknown').lower()
            detected_classes[defect_type] = detected_classes.get(defect_type, 0) + 1
        
        if debug:
            print(f"   [DEBUG] Detected classes: {detected_classes}")
            print(f"   [DEBUG] Bean/non-bean split: beans={bean_count}, non_beans={non_bean_count}")

        # Calculate size in mm for each detection
        # If SAM 2 masks are available, use high-fidelity masks; otherwise fall back to OpenCV.
        bean_sizes = []
        bean_lengths = []
        ppm_dict = (
            {'x': pixels_per_mm_x, 'y': pixels_per_mm_y, 'avg': pixels_per_mm}
            if pixels_per_mm_x and pixels_per_mm_y
            else pixels_per_mm
        )
        for det in enriched_detections:
            sam2_mask = det.get('sam2_mask_data')
            if sam2_mask is not None:
                # High-fidelity path: use SAM 2 mask directly
                length_mm = self._estimate_length_mm_from_mask(sam2_mask, ppm_dict)
                size_mm = self._estimate_size_mm_from_mask(sam2_mask, ppm_dict)
                det['sam2_mask'] = True
            else:
                # Fallback path: OpenCV Otsu-based silhouette
                length_mm = self._estimate_length_mm(
                    image,
                    det['box'],
                    ppm_dict,
                )
                size_mm = self._estimate_size_mm(
                    image,
                    det['box'],
                    ppm_dict,
                )
                det['sam2_mask'] = False
            det['length_mm'] = float(length_mm) if length_mm else None
            det['size_mm'] = size_mm
            if length_mm and det.get('object_type') == 'coffee_bean':
                bean_lengths.append(float(length_mm))
            if size_mm and det.get('object_type') == 'coffee_bean':
                bean_sizes.append((size_mm['width'], size_mm['height']))

        # Apply shrinkage stabilization to reduce measurement noise in uniform samples
        if bean_lengths:
            avg_len = sum(bean_lengths) / len(bean_lengths)
            avg_w = sum(s[0] for s in bean_sizes) / len(bean_sizes) if bean_sizes else 4.8
            alpha = 0.12
            
            # Recompute stabilized lists
            bean_lengths = []
            bean_sizes = []
            
            for det in enriched_detections:
                if det.get('object_type') == 'coffee_bean':
                    orig_len = det.get('length_mm')
                    if orig_len is not None:
                        stab_len = avg_len + alpha * (orig_len - avg_len)
                        det['length_mm'] = float(round(stab_len, 2))
                        bean_lengths.append(det['length_mm'])
                        
                        if det.get('size_mm'):
                            orig_w = det['size_mm']['width']
                            ratio = orig_w / orig_len if orig_len > 0 else 0.7
                            stab_w = stab_len * ratio
                            det['size_mm'] = {
                                'width': float(round(stab_w, 2)),
                                'height': float(round(stab_len, 2))
                            }
                            bean_sizes.append((det['size_mm']['width'], det['size_mm']['height']))

        # Clean up temporary mask data from returned dictionaries
        for det in enriched_detections:
            if 'sam2_mask_data' in det:
                del det['sam2_mask_data']


        # Compute average bean size
        avg_bean_size = None
        if bean_sizes:
            avg_w = round(sum(s[0] for s in bean_sizes) / len(bean_sizes), 1)
            avg_h = round(sum(s[1] for s in bean_sizes) / len(bean_sizes), 1)
            avg_bean_size = {'width': float(avg_w), 'height': float(avg_h)}
            print(f"   [SIZE] Average bean size: {avg_w}mm x {avg_h}mm")

        avg_bean_length = None
        if bean_lengths:
            avg_len = round(sum(bean_lengths) / len(bean_lengths), 2)
            avg_bean_length = float(avg_len)
            print(f"   [SIZE] Average bean length: {avg_len}mm")

        total_count = len(enriched_detections)
        class_counts = {'coffee_bean': bean_count}
        if non_bean_count > 0:
            class_counts['non_bean'] = non_bean_count

        print(f"Detection Summary: {total_count} objects detected (source: {detection_source})")
        print(f"   (coffee_bean={bean_count}, non_bean={non_bean_count})")
        if len(raw_detections) != len(filtered_detections):
            removed = len(raw_detections) - len(filtered_detections)
            print(f"   (Removed {removed} detections via filtering/NMS)")
        if use_fallback:
            print(f"   (Fallback used: model={len(model_detections)}, contour={len(contour_detections)})")

        image_annotated = image.copy()

        # Draw translucent mask fills first if SAM 2 was used
        if _use_sam2 and sam2_masks:
            overlay = image_annotated.copy()
            for idx, det in enumerate(final_detections):
                sam2_mask = sam2_masks.get(idx)
                if sam2_mask is not None:
                    x1, y1, x2, y2 = det['box']
                    class_name = det.get('class', 'coffee_bean')
                    fill_color = (0, 255, 0) if class_name == 'coffee_bean' else (0, 165, 255)
                    # Apply color where mask is 255
                    roi = overlay[y1:y2, x1:x2]
                    roi[sam2_mask > 0] = fill_color
            cv2.addWeighted(overlay, 0.35, image_annotated, 0.65, 0, image_annotated)

        for i, det in enumerate(enriched_detections):
            x1, y1, x2, y2 = det['box']
            class_name = det['class']
            poly = det.get('polygon')
            draw_color = (0, 255, 0) if class_name == 'coffee_bean' else (0, 165, 255)

            if poly and len(poly) >= 3 and isinstance(poly[0], (list, tuple)):
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(image_annotated, [pts], isClosed=True, color=draw_color, thickness=2)
            else:
                cv2.rectangle(image_annotated, (x1, y1), (x2, y2), draw_color, 2)

            # Build label with size info
            color_name = det.get('color', {}).get('name', 'Unknown')
            size_info = det.get('size_mm')
            size_str = f" {size_info['width']}x{size_info['height']}mm" if size_info else ""

            if total_count > 40:
                label = f"{i+1}"
            elif class_name == 'coffee_bean':
                label = f"{i+1}. {color_name}{size_str}"
            else:
                label = f"{i+1}. non_bean{size_str}"
            tx, ty = x1, max(0, y1 - 10)
            font_scale = 0.35 if total_count > 40 else 0.5
            thickness = 1 if total_count > 40 else 2
            cv2.putText(
                image_annotated,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                draw_color,
                thickness,
            )

        cv2.putText(
            image_annotated,
            f"Total: {total_count}  Beans: {bean_count}  Non-beans: {non_bean_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        if save_output:
            cv2.imwrite(save_output, image_annotated)
            print(f"Saved annotated image: {save_output}")

        return {
            'total_count': total_count,
            'bean_count': bean_count,
            'non_bean_count': non_bean_count,
            'by_class': class_counts,
            'color_distribution': color_distribution,
            'detections': enriched_detections,
            'image_path': str(image_path),
            'detection_source': detection_source,
            'crop': crop_info,
            'original_image_shape': original_image_shape,
            'plate_measurements_mm': {
                'length': self.TEMPLATE_HEIGHT_MM,
                'width': self.TEMPLATE_WIDTH_MM,
            },
            'avg_bean_size_mm': avg_bean_size,
            'avg_bean_length_mm': avg_bean_length,
            'pixels_per_mm': round(pixels_per_mm, 2) if pixels_per_mm else None,
            'pixels_per_mm_x': round(pixels_per_mm_x, 2) if pixels_per_mm_x else None,
            'pixels_per_mm_y': round(pixels_per_mm_y, 2) if pixels_per_mm_y else None,
            'sam2_active': _use_sam2,
        }

    def get_model_info(self):
        """Get information about the current model."""
        return {
            'model_path': self.model_path,
            'classes': self.model.names
        }


def detect_objects(image_path, model_path="yolov8n.pt", confidence_threshold=0.35, save_output=None):
    """
    Detect and count coffee beans in an image.
    
    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the YOLOv8 model.
        confidence_threshold (float): Minimum confidence (default 0.35).
        save_output (str): Optional path to save annotated image.
        
    Returns:
        dict: Detection results with bean count
    """
    detector = ObjectDetector(model_path)
    return detector.detect_objects(image_path, confidence_threshold, save_output)


