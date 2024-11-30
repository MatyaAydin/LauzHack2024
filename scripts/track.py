import cv2
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import numpy as np

# Implementing SORT
class Sort:
    def __init__(self):
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        updated_tracks = []
        for tracker in self.trackers:
            tracker.predict()
            if len(detections) > 0:
                distances = [np.linalg.norm(detection[:2] - tracker.x[:2]) for detection in detections]
                closest_idx = np.argmin(distances)
                if distances[closest_idx] < 50:  # Threshold for matching
                    tracker.update(detections[closest_idx])
                    updated_tracks.append(tracker)
                    detections.pop(closest_idx)
        self.trackers = updated_tracks + [KalmanBoxTracker(det) for det in detections]
        return [tracker.get_state() for tracker in self.trackers]


class KalmanBoxTracker:
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.x[:4] = bbox.reshape(-1, 1)

    def predict(self):
        self.kf.predict()

    def update(self, bbox):
        self.kf.update(bbox.reshape(-1, 1))

    def get_state(self):
        return self.kf.x[:4].flatten()


# Initialize SORT
tracker = Sort()

# Load YOLOv8
model = YOLO('yolov8s.pt')

# Input and output paths
input_video_path = "./data/live_feed_5min2.mp4"
output_video_path = "boat_tracked.mp4"

# Open video
cap = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects with YOLOv8
    results = model.predict(source=frame, conf=0.5, device="cpu")  # Use CPU explicitly
    detections = []
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            conf = box[4]
            cls = int(box[5])  # Class index
            if cls == 8:  # Class index for boats
                detections.append(np.array([x1, y1, x2, y2, conf]))

    # Update SORT tracker with detections
    tracked_objects = tracker.update(detections)

    # Draw tracked objects
    for obj in tracked_objects:
        x1, y1, x2, y2 = map(int, obj[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "Tracked Boat", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Write frame to output
    out.write(frame)

cap.release()
out.release()
print("Boat tracking with SORT completed and video saved.")
