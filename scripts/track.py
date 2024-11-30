import cv2
import torch
import numpy as np


import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sort.sort import Sort


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  


input_video_path = "./data/live_feed_5min2.mp4"
output_video_path = "./outputs/boat_tracked.mp4"

# Path to the binary mask
mask_path = "./data/background/clean_frame_mask.jpg"


mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise FileNotFoundError("Mask file not found. Check the path to 'clean_frame_mask.png'")


cap = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))


tracker = Sort()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)


    results = model(masked_frame)
    detections = results.xyxy[0]

    # Filter for boats (class 8 in YOLO)
    boat_detections = []
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        if int(cls) == 8:  # Class 8 corresponds to boats
            boat_detections.append([x1, y1, x2, y2, conf])


    boat_detections = np.array(boat_detections)


    tracked_objects = tracker.update(boat_detections)


    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    out.write(frame)

cap.release()
out.release()
print("Boat tracking completed and video saved.")
