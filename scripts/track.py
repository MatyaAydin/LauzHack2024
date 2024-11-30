import cv2
import numpy as np
import csv
import sys
import os
from ultralytics import YOLO

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sort.sort import Sort

# Initialize the YOLOv8 model
model = YOLO('yolov8s.pt')  # Ensure you have the YOLOv8 weights in this path

# File paths
input_video_path = "./data/live_feed_5min2.mp4"
output_video_path = "./outputs/boat_tracked.mp4"
mask_path = "./data/background/clean_frame_mask.jpg"
csv_output_path = "./outputs/boat_tracking_log.csv"

# Load the mask
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise FileNotFoundError("Mask file not found. Check the path to 'clean_frame_mask.jpg'")

# Initialize video capture and output
cap = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

tracker = Sort()

# Object tracking data
tracking_log = {}
frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Increment frame count
    frame_count += 1
    timestamp = frame_count / frame_rate  # Current time in seconds

    # Apply mask to the frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Run YOLOv8 detection
    results = model(masked_frame)
    detections = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
    classes = results[0].boxes.cls.cpu().numpy()  # Get classes
    confidences = results[0].boxes.conf.cpu().numpy()  # Get confidences

    # Filter for boats (class 8 in YOLO)
    # Filter for boats (class 8 in YOLO)
    boat_detections = []
    for i, cls in enumerate(classes):
        if int(cls) == 8:  # Class 8 corresponds to boats
            x1, y1, x2, y2 = detections[i][:4]
            conf = confidences[i]
            boat_detections.append([x1, y1, x2, y2, conf])

    # Ensure boat_detections is a NumPy array with the correct shape
    boat_detections = np.array(boat_detections)
    if boat_detections.size == 0:
        boat_detections = np.empty((0, 5))  # Empty array with 5 columns

    # Update tracker with detections
    tracked_objects = tracker.update(boat_detections)


    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj.astype(int)

        # Record timestamps
        if obj_id not in tracking_log:
            tracking_log[obj_id] = {'in': timestamp, 'out': None}
        else:
            tracking_log[obj_id]['out'] = timestamp

        # Draw bounding boxes and ID on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Write the frame to output video
    out.write(frame)

cap.release()
out.release()

# Filter and write to CSV
with open(csv_output_path, 'w', newline='') as csvfile:
    fieldnames = ['ID', 'Timestamp In', 'Timestamp Out', 'Duration (s)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for obj_id, timestamps in tracking_log.items():
        if timestamps['out'] is not None:
            duration = timestamps['out'] - timestamps['in']
            if duration >= 10:  # Only log objects visible for 10 seconds or more
                writer.writerow({
                    'ID': obj_id,
                    'Timestamp In': round(timestamps['in'], 2),
                    'Timestamp Out': round(timestamps['out'], 2),
                    'Duration (s)': round(duration, 2)
                })

print("Boat tracking completed, video saved, and CSV log generated.")
