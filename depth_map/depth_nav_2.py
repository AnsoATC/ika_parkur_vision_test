#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2
import os
from time import strftime, gmtime

def detect_navigable_zone(depth_image, width, height):
    """
    Detect navigable zone by thresholding depth to find floor (free space).
    Returns: (x, y, w, h) of rectangle, largest_contour for navigable zone.
    """
    # Threshold for floor (depth > 3m, beyond barriers)
    floor_mask = cv2.inRange(depth_image, 3000, 10000)  # 3m to 10m
    # Morphological operations to clean noise
    kernel = np.ones((5, 5), np.uint8)
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel)
    
    # Find largest contour (assumed to be floor)
    contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 1000:
            x, y, w, h = cv2.boundingRect(largest_contour)
            # Center rectangle in navigable zone
            center_x = x + w // 2
            return (center_x - w//4, height//4, w//2, height//2), largest_contour
    return (width//4, height//4, width//2, height//2), None  # Fallback: center rectangle, no contour

def main():
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    pipeline.start(config)

    # Create output directory
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Colorize depth for visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Detect navigable zone
            (x, y, w, h), largest_contour = detect_navigable_zone(depth_image, 1280, 720)
            annotated_image = color_image.copy()
            # Draw contour of navigable zone (blue)
            if largest_contour is not None:
                cv2.drawContours(annotated_image, [largest_contour], -1, (255, 0, 0), 2)
            # Draw rectangle (green)
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated_image, 'Navigable Zone', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display frames
            cv2.imshow('RGB Frame', color_image)
            cv2.imshow('Depth Frame', depth_image)
            cv2.imshow('Depth Heatmap', depth_colormap)
            cv2.imshow('Navigable Zone', annotated_image)

            # Save frames on 'p' key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                timestamp = strftime('%Y%m%d_%H%M%S', gmtime())
                cv2.imwrite(os.path.join(output_dir, f'rgb_{timestamp}.jpg'), color_image)
                cv2.imwrite(os.path.join(output_dir, f'depth_{timestamp}.jpg'), depth_colormap)
                cv2.imwrite(os.path.join(output_dir, f'zone_{timestamp}.jpg'), annotated_image)
                print(f"Saved images with timestamp {timestamp}")

            # Exit on 'q'
            if key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
