"""
Module: cvat_converter.py
Description: Converts CVAT XML annotations to YOLO format.
Parses polygons, extracts bounding boxes, and generates .txt label files.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import re


class CVATtoYOLOConverter:
    """Convert CVAT XML annotations to YOLO txt format."""
    
    def __init__(self, class_map=None):
        """
        Initialize converter with optional class name mapping.
        
        Args:
            class_map (dict): Map CVAT class names to YOLO class IDs.
                             e.g., {'bean': 0, 'defect': 1}
                             Default: auto-detect from XML.
        """
        self.class_map = class_map or {}
        self.reverse_map = {}  # {class_name: class_id}
    
    def parse_cvat_xml(self, xml_path):
        """
        Parse CVAT XML and extract image/annotation data.
        
        Args:
            xml_path (str): Path to annotations.xml from CVAT export.
            
        Returns:
            dict: {image_name: {'width': int, 'height': int, 'objects': [...]}}
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        images = {}
        
        # Extract class labels first
        labels = root.findall(".//labels/label")
        for idx, label in enumerate(labels):
            class_name = label.find("name").text
            if class_name not in self.reverse_map:
                self.reverse_map[class_name] = idx
        
        # Extract images and annotations
        for image in root.findall(".//image"):
            img_name = image.get("name")
            width = int(image.get("width"))
            height = int(image.get("height"))
            
            objects = []
            
            # Parse bounding boxes
            for bbox in image.findall(".//box"):
                class_name = bbox.get("label")
                x1 = float(bbox.get("xtl"))
                y1 = float(bbox.get("ytl"))
                x2 = float(bbox.get("xbr"))
                y2 = float(bbox.get("ybr"))
                
                objects.append({
                    'type': 'box',
                    'class': class_name,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                })
            
            # Parse polygons (convert to bounding box)
            for polygon in image.findall(".//polygon"):
                class_name = polygon.get("label")
                points_str = polygon.get("points")
                
                # Parse points: "x1,y1;x2,y2;..."
                points = []
                if points_str:
                    for pt in points_str.split(";"):
                        if pt.strip():
                            x, y = map(float, pt.strip().split(","))
                            points.append((x, y))
                
                if points:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
                    
                    objects.append({
                        'type': 'polygon',
                        'class': class_name,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                    })
            
            images[img_name] = {
                'width': width,
                'height': height,
                'objects': objects
            }
        
        return images
    
    def convert_to_yolo(self, images_data, class_name_map=None):
        """
        Convert extracted image data to YOLO format.
        
        Args:
            images_data (dict): Output from parse_cvat_xml().
            class_name_map (dict): Map CVAT class names to YOLO IDs.
                                   e.g., {'bean': 0}
                                   If None, uses auto-detected class_map.
            
            Returns:
            dict: {image_name: [yolo_lines_str]}
                 Each line: "class_id x_center y_center width height" (normalized 0..1)
        """
        if class_name_map:
            self.reverse_map = class_name_map
        
        yolo_data = {}
        
        for img_name, img_info in images_data.items():
            width = img_info['width']
            height = img_info['height']
            
            yolo_lines = []
            
            for obj in img_info['objects']:
                class_name = obj['class']
                
                # Get class ID
                if class_name not in self.reverse_map:
                    # Auto-assign if not in map
                    class_id = len(self.reverse_map)
                    self.reverse_map[class_name] = class_id
                else:
                    class_id = self.reverse_map[class_name]
                
                # Extract bbox
                x1, y1 = obj['x1'], obj['y1']
                x2, y2 = obj['x2'], obj['y2']
                
                # Convert to YOLO format (center + normalized width/height)
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                bbox_width = (x2 - x1) / width
                bbox_height = (y2 - y1) / height
                
                # Clamp to [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                bbox_width = max(0, min(1, bbox_width))
                bbox_height = max(0, min(1, bbox_height))
                
                line = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
                yolo_lines.append(line)
            
            yolo_data[img_name] = yolo_lines
        
        return yolo_data
    
    def save_labels(self, yolo_data, output_dir, replace_ext=True):
        """
        Save YOLO labels to txt files.
        
        Args:
            yolo_data (dict): Output from convert_to_yolo().
            output_dir (str): Directory to save .txt files.
            replace_ext (bool): Replace image extension with .txt.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for img_name, lines in yolo_data.items():
            # Replace image extension with .txt
            if replace_ext:
                label_name = Path(img_name).stem + ".txt"
            else:
                label_name = img_name + ".txt"
            
            label_path = output_path / label_name
            
            if lines:
                with open(label_path, 'w') as f:
                    f.write("\n".join(lines))
                print(f"✅ Saved: {label_path}")
            else:
                # Create empty file for images with no annotations
                label_path.touch()
                print(f"⚠️  Empty labels: {label_path}")
    
    def get_class_map(self):
        """Return the detected/used class-to-ID mapping."""
        return self.reverse_map


def convert_cvat_to_yolo(cvat_xml_path, output_dir, class_name_map=None):
    """
    Simple function to convert CVAT XML to YOLO labels.
    
    Args:
        cvat_xml_path (str): Path to CVAT annotations.xml.
        output_dir (str): Directory to save .txt label files.
        class_name_map (dict): Optional class name → class ID mapping.
        
    Returns:
        dict: Class-to-ID mapping used.
    """
    converter = CVATtoYOLOConverter(class_map=class_name_map)
    images_data = converter.parse_cvat_xml(cvat_xml_path)
    yolo_data = converter.convert_to_yolo(images_data, class_name_map)
    converter.save_labels(yolo_data, output_dir)
    
    print(f"\n📊 Conversion Summary:")
    print(f"   Images: {len(yolo_data)}")
    print(f"   Class map: {converter.get_class_map()}")
    
    return converter.get_class_map()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python cvat_converter.py <cvat_xml_path> <output_dir> [class_map_json]")
        sys.exit(1)
    
    cvat_xml_path = sys.argv[1]
    output_dir = sys.argv[2]
    class_map = None
    
    if len(sys.argv) > 3:
        import json
        class_map = json.loads(sys.argv[3])
    
    convert_cvat_to_yolo(cvat_xml_path, output_dir, class_map)
