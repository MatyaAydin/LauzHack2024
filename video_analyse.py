
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os
from sort.sort import Sort


def get_count_video(video_path):
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    # Load the YOLOv8 model (use 'yolov8n.pt' for faster performance)
    model = YOLO("yolov8n.pt")  # Replace with 'yolov8s.pt', etc., for different model sizes

    # Initialize tracker
    tracker = Sort()

    count = 0
    tracked_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)

        detections = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls)
                score = box.conf
                # COCO class for 'boat' is 9
                if cls == 9 and score > 0.5:
                    x1, y1, x2, y2 = box.xyxy[0]
                    detections.append([x1.item(), y1.item(), x2.item(), y2.item(), score.item()])

        if detections:
            dets = np.array(detections)
            tracked_objects = tracker.update(dets)

            for obj in tracked_objects:
                obj_id = int(obj[4])
                if obj_id not in tracked_ids:
                    tracked_ids.add(obj_id)
                    count += 1

    cap.release()
    return count