import cv2
import torch
import numpy as np

# Load YOLO model (YOLOv5 example)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use YOLOv5s pretrained weights

# Input and output video paths
input_video_path = "./data/live_feed_5min2.mp4"
output_video_path = "boat_detected2.mp4"

# Path to the binary mask
mask_path = "./data/background/clean_frame2.jpg"

# Load the binary mask
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise FileNotFoundError("Mask file not found. Check the path to 'clean_frame_mask.png'")

# Open video
cap = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the mask to the frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Apply the mask to the frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # # Save or display the masked frame to verify
    # cv2.imshow("Masked Frame", masked_frame)  # Show the masked frame
    # cv2.imwrite("masked_frame_sample.jpg", masked_frame)  # Save a sample frame

    # # Wait for a key press to proceed (optional for debugging)
    # cv2.waitKey(0)


    # Run YOLO detection on the masked frame
    results = model(masked_frame)
    detections = results.xyxy[0]  # [x1, y1, x2, y2, confidence, class]

    # Draw bounding boxes
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        if int(cls) == 8:  # Class 8 corresponds to boats in YOLO
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Boat: {conf:.2f}", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write to output
    out.write(frame)

cap.release()
out.release()
print("Boat detection completed and video saved.")
