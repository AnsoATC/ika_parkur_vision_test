#!/usr/bin/env python3
import cv2
import numpy as np
import os
from ultralytics import YOLO
from time import strftime, gmtime

def detect_objects():
    """
    Detect traffic barriers, cones, and signs in an image using YOLOv12n.
    Display and save annotated image.
    """
    # Hardcode input image and model paths
    input_image = "../input_images/rgb_20250704_201617.jpg"
    model_path = "models/yolo12n_detection.pt"
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

    # Perform detection
    results = model.predict(image, conf=0.5, iou=0.5, imgsz=640)

    # Draw bounding boxes and labels
    annotated_image = image.copy()
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        classes = result.boxes.cls.cpu().numpy()  # Class IDs
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_names = result.names  # Class names dictionary

        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            label = f"{class_names[int(cls)]} {conf:.2f}"
            color = (0, 255, 0) if class_names[int(cls)] == "traffic_barrier" else \
                    (0, 0, 255) if class_names[int(cls)] == "traffic_cone" else (255, 0, 0)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display result
    cv2.imshow('Detected Objects', annotated_image)

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    timestamp = strftime('%Y%m%d_%H%M%S', gmtime())
    output_path = os.path.join(output_dir, f'detected_{timestamp}.jpg')
    cv2.imwrite(output_path, annotated_image)
    print(f"Saved annotated image to {output_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_objects()
