#!/usr/bin/env python3
import cv2
import numpy as np
import os
from ultralytics import YOLO
from time import strftime, gmtime

def segment_road():
    """
    Segment navigable road in an image using YOLOv11n.
    Display and save image with road mask overlaid.
    """
    # Hardcode input image and model paths
    input_image = "../input_images/rgb_20250704_201617.jpg"
    model_path = "models/yolo11n_segmentation.pt"
    output_dir = "../output_images"

    # Load model
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found.")
        return
    model = YOLO(model_path)

    # Load image
    if not os.path.exists(input_image):
        print(f"Error: Image {input_image} not found.")
        return
    image = cv2.imread(input_image)
    if image is None:
        print(f"Error: Failed to load image {input_image}.")
        return

    # Perform segmentation
    results = model.predict(image, conf=0.5, iou=0.5, imgsz=640)

    # Create mask and overlay
    annotated_image = image.copy()
    for result in results:
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()  # Segmentation masks
            for mask in masks:
                # Resize mask to match image dimensions
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                # Create green overlay for navigable road
                green_mask = np.zeros_like(image)
                green_mask[mask > 0] = [0, 255, 0]  # Green for navigable road
                # Blend mask with image
                annotated_image = cv2.addWeighted(annotated_image, 0.7, green_mask, 0.3, 0)

    # Display result
    cv2.imshow('Segmented Road', annotated_image)

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    timestamp = strftime('%Y%m%d_%H%M%S', gmtime())
    output_path = os.path.join(output_dir, f'segmented_{timestamp}.jpg')
    cv2.imwrite(output_path, annotated_image)
    print(f"Saved segmented image to {output_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    segment_road()
