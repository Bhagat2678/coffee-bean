"""
Module: detector.py
Description: Detects and counts coffee beans using classic OpenCV (No YOLO).
"""

import cv2
import numpy as np

def count_beans_opencv(image_path, output_path=None):
    """
    Counts beans in an image using thresholding and contour detection.
    
    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the image with counted beans drawn.
        
    Returns:
        int: The number of beans detected.
    """
    
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return 0

    # 2. Preprocessing
    # Convert to grayscale (Color is not needed for counting, just shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur to remove noise (dust, texture details)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 3. Thresholding (The most important step)
    # We use Otsu's method to automatically find the best separation between
    # dark beans and light background.
    # If you have a dark background, remove 'cv2.THRESH_BINARY_INV'.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Morphological Operations (Cleaning)
    # This separates beans that are slightly touching and removes small noise.
    kernel = np.ones((3, 3), np.uint8)
    clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 5. Find Contours
    # This finds the boundaries of the white blobs in the black/white mask.
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bean_count = 0
    min_bean_area = 100  # Filter out small specks/dust (Adjust this if needed)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area > min_bean_area:
            bean_count += 1
                       
            # Calculate the center of the bean
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Draw the contour outline (Green)
                cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
                # Put the count number on the bean
                cv2.putText(img, str(bean_count), (cX - 10, cY), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 6. Save or Show Result
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Processed image saved to: {output_path}")

    return bean_count