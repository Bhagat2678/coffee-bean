"""
Module: analyzer.py
Description: Enhanced analysis of detected coffee beans - color, size, and count.
"""

import cv2
import numpy as np
from pathlib import Path
from src.detector import ObjectDetector


class BeanAnalyzer:
    """Analyze detected beans for color, size, and other properties."""
    
    def __init__(self, detector=None):
        """
        Initialize analyzer with optional detector.
        
        Args:
            detector (ObjectDetector): YOLOv8 detector instance.
            If None, creates default nano model.
        """
        self.detector = detector or ObjectDetector("yolov8n.pt")
    
    @staticmethod
    def get_bean_color(image, bbox, method="dominant"):
        """
        Extract color information from bean bounding box.
        
        Args:
            image (np.ndarray): Image array (BGR).
            bbox (list): Bounding box [x1, y1, x2, y2].
            method (str): 'dominant' (most common), 'average', or 'histogram'.
            
        Returns:
            dict: Color information including RGB, HSV, color name.
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return {'error': 'Empty ROI'}
        
        if method == "dominant":
            # Find most common color
            pixels = roi.reshape(-1, 3)
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
            dominant_idx = np.argmax(counts)
            bgr = unique_colors[dominant_idx]
        elif method == "average":
            # Average color
            bgr = np.mean(roi, axis=(0, 1)).astype(int)
        else:  # histogram (weighted by frequency)
            pixels = roi.reshape(-1, 3)
            bgr = np.median(pixels, axis=0).astype(int)
        
        # Convert BGR to RGB for readable output
        rgb = [int(bgr[2]), int(bgr[1]), int(bgr[0])]
        
        # Convert BGR to HSV for hue-based color naming
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h = np.mean(roi_hsv[:, :, 0])
        s = np.mean(roi_hsv[:, :, 1])
        v = np.mean(roi_hsv[:, :, 2])
        
        # Color name based on HSV
        color_name = BeanAnalyzer._get_color_name(h, s, v)
        
        return {
            'rgb': rgb,
            'bgr': [int(x) for x in bgr],
            'hsv': {'h': float(h), 's': float(s), 'v': float(v)},
            'color_name': color_name,
            'hex': f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        }
    
    @staticmethod
    def _get_color_name(h, s, v):
        """Map HSV to human-readable color name."""
        # If brightness is very low, it's black
        if v < 50:
            return "Black"
        
        # If saturation is very low, it's grayscale
        if s < 30:
            return "Gray" if v < 150 else "White"
        
        # Map hue to color name (0-180 in OpenCV HSV)
        if h < 10 or h > 170:
            return "Red"
        elif h < 30:
            return "Orange-Brown"
        elif h < 50:
            return "Brown"
        elif h < 70:
            return "Yellow-Brown"
        elif h < 100:
            return "Green"
        elif h < 120:
            return "Cyan"
        elif h < 140:
            return "Blue"
        elif h < 160:
            return "Purple"
        else:
            return "Magenta"
    
    @staticmethod
    def get_bean_size(bbox, image_shape=None):
        """
        Calculate bean size from bounding box.
        
        Args:
            bbox (list): Bounding box [x1, y1, x2, y2].
            image_shape (tuple): Image shape (height, width, channels) for reference.
            
        Returns:
            dict: Size metrics - width, height, area, relative_area.
        """
        x1, y1, x2, y2 = bbox
        
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        result = {
            'width_px': float(width),
            'height_px': float(height),
            'area_px': float(area),
            'aspect_ratio': float(width / height) if height > 0 else 0
        }
        
        # Calculate relative size if image shape provided
        if image_shape:
            img_area = image_shape[0] * image_shape[1]
            result['relative_area_percent'] = round((area / img_area) * 100, 2)
        
        # Classify size
        if area < 1000:
            result['size_class'] = "Tiny"
        elif area < 5000:
            result['size_class'] = "Small"
        elif area < 15000:
            result['size_class'] = "Medium"
        elif area < 30000:
            result['size_class'] = "Large"
        else:
            result['size_class'] = "Very Large"
        
        return result
    
    def analyze_image(self, image_path, confidence_threshold=0.45, save_output=None):
        """
        Comprehensive analysis of beans in image.
        
        Args:
            image_path (str): Path to image.
            confidence_threshold (float): Detection confidence threshold.
            save_output (str): Path to save annotated image.
            
        Returns:
            dict: Detailed analysis results.
        """
        image = cv2.imread(image_path)
        if image is None:
            return {'error': f'Cannot load image: {image_path}'}
        
        # Detect beans
        detection_results = self.detector.detect_objects(
            image_path, 
            confidence_threshold, 
            save_output=None  # We'll annotate ourselves
        )
        
        # Enhance with color and size analysis
        beans = []
        for i, detection in enumerate(detection_results['detections']):
            bbox = detection['box']
            color_info = self.get_bean_color(image, bbox)
            size_info = self.get_bean_size(bbox, image.shape)
            
            beans.append({
                'id': i + 1,
                'class': detection['class'],
                'confidence': detection['confidence'],
                'bbox': bbox,
                'color': color_info,
                'size': size_info
            })
        
        # Draw annotated image with color info
        if save_output:
            annotated = image.copy()
            for bean in beans:
                x1, y1, x2, y2 = map(int, bean['bbox'])
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Label with ID and color
                color_name = bean['color'].get('color_name', 'Unknown')
                size_class = bean['size']['size_class']
                label = f"Bean {bean['id']}: {color_name} ({size_class})"
                
                cv2.putText(annotated, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if save_output:
                cv2.imwrite(save_output, annotated)
                print(f"ðŸ“¸ Annotated image saved: {save_output}")
        
        return {
            'image_path': str(image_path),
            'total_beans': len(beans),
            'beans': beans,
            'summary': {
                'total_count': detection_results['total_count'],
                'by_class': detection_results['by_class'],
                'color_distribution': self._get_color_distribution(beans),
                'size_distribution': self._get_size_distribution(beans)
            }
        }
    
    @staticmethod
    def _get_color_distribution(beans):
        """Get distribution of bean colors."""
        color_counts = {}
        for bean in beans:
            color_name = bean['color'].get('color_name', 'Unknown')
            color_counts[color_name] = color_counts.get(color_name, 0) + 1
        return color_counts
    
    @staticmethod
    def _get_size_distribution(beans):
        """Get distribution of bean sizes."""
        size_counts = {}
        for bean in beans:
            size_class = bean['size']['size_class']
            size_counts[size_class] = size_counts.get(size_class, 0) + 1
        return size_counts


def analyze_beans(image_path, model_path="yolov8n.pt", save_output=None):
    """
    Simple function to analyze beans in an image.
    
    Args:
        image_path (str): Path to image.
        model_path (str): Path to trained YOLOv8 model.
        save_output (str): Path to save annotated result.
        
    Returns:
        dict: Analysis results with color, size, and count.
    """
    detector = ObjectDetector(model_path)
    analyzer = BeanAnalyzer(detector)
    
    return analyzer.analyze_image(image_path, save_output=save_output)


if __name__ == "__main__":
    import json
    
    # Example usage
    TEST_IMAGE = "data/raw/test_beans_1.jpg"
    MODEL_PATH = "runs/detect/runs/coffee_beans/detection_v1/weights/best.pt"
    OUTPUT_IMAGE = "data/output/analyzed_beans.jpg"
    
    if Path(TEST_IMAGE).exists():
        detector = ObjectDetector(MODEL_PATH)
        analyzer = BeanAnalyzer(detector)
        
        results = analyzer.analyze_image(TEST_IMAGE, save_output=OUTPUT_IMAGE)
        
        print("\n" + "="*70)
        print("â˜• BEAN ANALYSIS RESULTS")
        print("="*70)
        print(json.dumps(results['summary'], indent=2))
        print("\nðŸ“Š Individual Bean Details:")
        for bean in results['beans']:
            print(f"\n  Bean {bean['id']}:")
            print(f"    - Color: {bean['color']['color_name']} (RGB: {bean['color']['rgb']})")
            print(f"    - Size: {bean['size']['size_class']} ({bean['size']['width_px']:.0f}x{bean['size']['height_px']:.0f}px)")
            print(f"    - Confidence: {bean['confidence']:.2%}")
    else:
        print(f"âŒ Test image not found: {TEST_IMAGE}")

