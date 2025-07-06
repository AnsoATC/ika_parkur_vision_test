#!/usr/bin/env python3
import cv2
import numpy as np
import os

def segment_barriers():
    """
    Segment white and red barriers in an image, display masks and masked image.
    """
    # Hardcode input image path
    input_image = "input_images/rgb_20250704_191441.jpg"
    output_dir = "output_images"

    # Load image
    if not os.path.exists(input_image):
        print(f"Error: Image {input_image} not found.")
        return
    image = cv2.imread(input_image)
    if image is None:
        print(f"Error: Failed to load image {input_image}.")
        return

    # Convert to HSV for robust color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for white and red
    white_lower = np.array([0, 0, 220])
    white_upper = np.array([180, 50, 255])
    red_lower1 = np.array([0, 150, 100])
    red_upper1 = np.array([8, 255, 255])
    red_lower2 = np.array([170, 150, 100])
    red_upper2 = np.array([180, 255, 255])

    # Create masks
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    combined_mask = cv2.bitwise_or(white_mask, red_mask)

    # Morphological operations to enhance barrier lines
    kernel = np.ones((7, 7), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # Create masked image (only white/red regions visible)
    masked_image = cv2.bitwise_and(image, image, mask=combined_mask)

    # Find contours for debugging (optional bounding boxes)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated_image = image.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # Filter small contours
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display results
    cv2.imshow('Original Image', image)
    cv2.imshow('White Mask', white_mask)
    cv2.imshow('Red Mask', red_mask)
    cv2.imshow('Masked Image', masked_image)
    cv2.imshow('Annotated Image', annotated_image)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_image))[0]
    cv2.imwrite(os.path.join(output_dir, f'{base_name}_white_mask.jpg'), white_mask)
    cv2.imwrite(os.path.join(output_dir, f'{base_name}_red_mask.jpg'), red_mask)
    cv2.imwrite(os.path.join(output_dir, f'{base_name}_masked.jpg'), masked_image)
    cv2.imwrite(os.path.join(output_dir, f'{base_name}_annotated.jpg'), annotated_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    segment_barriers()
