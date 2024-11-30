from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')

# Predict on an image or video
results = model.predict(source='./data/live_feed_5min.mp4', save=True)

# Access bounding box coordinates
for result in results:
    boxes = result.boxes  # Get bounding box data
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
        conf = box.conf[0]  # Confidence score
        cls = box.cls[0]  # Class index
        print(f"Bounding Box: ({x1}, {y1}, {x2}, {y2}), Confidence: {conf}, Class: {cls}")
