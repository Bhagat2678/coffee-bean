"""
Module: detector.py
Description: Detects and counts objects in images using YOLOv8.
Supports both pre-trained models and custom trained models.
Includes robust NMS post-processing to handle overlapping detections.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


class ObjectDetector:
    """
    YOLOv8-based object detector for counting and classifying objects in images.
    """
    
    def __init__(self, model_path="yolov8n.pt"):
        """
        Initialize the detector with a YOLOv8 model.
        
        Args:
            model_path (str): Path to the YOLOv8 model file.
        """
        self.model = YOLO(model_path)
        self.model_path = model_path
        print(f"[INFO] Model loaded: {model_path}")
    
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
    
    def _is_likely_coin(self, box, color_info, image_shape, all_boxes=None):
        """
        Detect if a bounding box likely contains a coin rather than a coffee bean.
        Uses shape (round/square), color (metallic), and relative size heuristics.
        """
        x1, y1, x2, y2 = map(int, box)
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        aspect = float(bw) / float(bh)

        s_mean = float(color_info['hsv']['s'])
        v_mean = float(color_info['hsv']['v'])
        h_mean = float(color_info['hsv']['h'])
        gray_std = float(color_info.get('gray_std', 999))
        box_area = float(bw * bh)

        # Coins are round → near-square bounding box (aspect ratio 0.7 - 1.4)
        is_squarish = 0.7 <= aspect <= 1.4

        # Coins are metallic: low saturation, moderate-to-high brightness
        is_metallic = (s_mean < 60 and v_mean > 80)

        # Coins have uniform texture (low standard deviation in grayscale)
        is_uniform = gray_std < 40

        # Coins are typically larger than individual beans
        is_larger = False
        if all_boxes and len(all_boxes) > 2:
            areas = [float(max(1, b[2]-b[0]) * max(1, b[3]-b[1])) for b in all_boxes]
            median_area = sorted(areas)[len(areas) // 2]
            if box_area > median_area * 1.5:
                is_larger = True

        # Need at least squarish shape + one of (metallic color, uniform texture, larger size)
        if is_squarish and (is_metallic or (is_uniform and is_larger)):
            return True

        # Strong metallic signal + uniform → coin even if not perfectly square
        if is_metallic and is_uniform and v_mean > 120 and s_mean < 30:
            return True

        return False

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
    
    def detect_objects(self, image_path, confidence_threshold=0.25, save_output=None,
                       iou=0.3,
                       min_confidence_output=0.25,
                       min_box_area_ratio=0.0005,
                       max_box_area_ratio=0.7,
                       min_aspect=0.25,
                       max_aspect=4.0,
                       box_shrink_ratio=0.15,
                       use_contour_fallback=False,
                       debug=False):
        """
        Detect objects in an image and count them by class with extra post-processing.

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
            debug (bool): If True, print per-detection filtering decisions.

        Returns:
            dict: Detection results with filtered counts and boxes.
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

        h, w = image.shape[:2]
        img_area = float(w * h)
        if debug:
            print(f"Image dims from cv2: h={h}, w={w}, img_area={img_area}")

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

            if d['confidence'] < min_confidence_output:
                if debug:
                    print(f"Discarding by conf {d['confidence']:.4f} < {min_confidence_output}")
                continue
            if area_ratio < min_box_area_ratio:
                if debug:
                    print(f"Discarding tiny box area_ratio={area_ratio:.6f}")
                continue
            if area_ratio > max_box_area_ratio:
                if debug:
                    print(f"Discarding large box area_ratio={area_ratio:.3f}")
                continue
            if aspect < min_aspect or aspect > max_aspect:
                if debug:
                    print(f"Discarding by aspect ratio={aspect:.3f}")
                continue

            d_out = d.copy()
            d_out['confidence'] = round(d_out['confidence'], 3)
            d_out['box'] = self._shrink_box(d_out['box'], shrink_ratio=box_shrink_ratio)
            prefiltered.append(d_out)

        filtered_detections = self._apply_nms(prefiltered, iou_threshold=iou)

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

        # Enrich detections with object type and color info.
        enriched_detections = []
        bean_count = 0
        non_bean_count = 0
        # Known model classes that are coffee beans (including defective beans)
        BEAN_CLASSES = {'black', 'broken', 'green', 'immature', 'infested', 'sour'}
        # Everything else (foreign, fraghusk, husk, coin, etc.) is non_bean

        color_distribution = {}
        all_boxes = [det['box'] for det in final_detections]
        for det in final_detections:
            det_copy = det.copy()
            color_info = self._extract_color_info(image, det_copy['box'])
            model_class = det.get('class', '').lower()

            # Check if detection looks like a coin (override model class)
            if self._is_likely_coin(det_copy['box'], color_info, image.shape, all_boxes):
                object_type = 'non_bean'
                model_class = 'coin'
            # Use model's class label to determine bean vs non-bean
            elif model_class in BEAN_CLASSES:
                object_type = 'coffee_bean'
            else:
                object_type = 'non_bean'

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

        # Final NMS pass to catch any remaining overlapping boxes.
        enriched_detections = self._apply_nms(enriched_detections, iou_threshold=iou)
        # Recount after final NMS
        bean_count = sum(1 for d in enriched_detections if d.get('object_type') == 'coffee_bean')
        non_bean_count = sum(1 for d in enriched_detections if d.get('object_type') != 'coffee_bean')

        # --- Size estimation using 5 rupee coin as reference (23mm diameter) ---
        COIN_DIAMETER_MM = 23.0
        pixels_per_mm = None

        # Find the coin detection to calibrate pixel-to-mm ratio
        for det in enriched_detections:
            if det.get('defect_type') == 'coin':
                x1, y1, x2, y2 = det['box']
                coin_w = max(1, x2 - x1)
                coin_h = max(1, y2 - y1)
                coin_pixel_diameter = (coin_w + coin_h) / 2.0  # average of width and height
                pixels_per_mm = coin_pixel_diameter / COIN_DIAMETER_MM
                print(f"   [SIZE] Coin found: {coin_w}x{coin_h}px, pixels_per_mm={pixels_per_mm:.2f}")
                break

        # Calculate size in mm for each detection
        bean_sizes = []
        for det in enriched_detections:
            x1, y1, x2, y2 = det['box']
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            if pixels_per_mm and pixels_per_mm > 0:
                width_mm = round(bw / pixels_per_mm, 1)
                height_mm = round(bh / pixels_per_mm, 1)
                det['size_mm'] = {'width': width_mm, 'height': height_mm}
                if det.get('object_type') == 'coffee_bean':
                    bean_sizes.append((width_mm, height_mm))
            else:
                det['size_mm'] = None

        # Compute average bean size
        avg_bean_size = None
        if bean_sizes:
            avg_w = round(sum(s[0] for s in bean_sizes) / len(bean_sizes), 1)
            avg_h = round(sum(s[1] for s in bean_sizes) / len(bean_sizes), 1)
            avg_bean_size = {'width': avg_w, 'height': avg_h}
            print(f"   [SIZE] Average bean size: {avg_w}mm x {avg_h}mm")

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
            elif det.get('defect_type') == 'coin':
                label = f"{i+1}. Coin (ref)"
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
            'avg_bean_size_mm': avg_bean_size,
            'pixels_per_mm': round(pixels_per_mm, 2) if pixels_per_mm else None,
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


