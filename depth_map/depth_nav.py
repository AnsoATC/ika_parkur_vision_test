#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2
import os
from time import strftime, gmtime

def detect_navigable_zone(depth_image, width, height):
    """
    Detect navigable zone between barriers using depth data.
    Returns: (x, y, w, h) of navigable zone rectangle.
    """
    # Assume barriers are within 0.5m-3m depth
    depth_mask = cv2.inRange(depth_image, 500, 3000)  # 0.5m to 3m
    # Find contours of barriers
    contours, _ = cv2.findContours(depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to find left and right barriers
    left_x, right_x = width, 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:  # Large contours only
            x, y, w, h = cv2.boundingRect(cnt)
            if x + w/2 < width/2:  # Left side
                left_x = min(left_x, x + w)
            else:  # Right side
                right_x = max(right_x, x)
    
    # Compute navigable zone (center between barriers)
    if left_x < width and right_x > 0:
        center_x = (left_x + right_x) // 2
        zone_width = right_x - left_x
        return center_x - zone_width//4, height//4, zone_width//2, height//2
    return width//4, height//4, width//2, height//2  # Fallback: center rectangle

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
            x, y, w, h = detect_navigable_zone(depth_image, 1280, 720)
            annotated_image = color_image.copy()
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
