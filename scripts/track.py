import os
import cv2
import numpy as np
import csv
import torch
import sys
import os
import random  # For generating random colors

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from sort.sort import Sort
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


class BoatTracker:
    def __init__(self, input_video_path, mask_path="./data/background/clean_frame_mask.jpg"):
        self.input_video_path = input_video_path
        self.mask_path = mask_path

        # Derive output paths relative to input video path
        base_name = os.path.splitext(os.path.basename(input_video_path))[0]
        output_dir = os.path.join(os.path.dirname(input_video_path), "output")
        os.makedirs(output_dir, exist_ok=True)

        self.output_video_path = os.path.join(output_dir, f"{base_name}_tracked.mp4")
        self.csv_output_path = os.path.join(output_dir, f"{base_name}_tracking_log.csv")

        # Initialize the YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use YOLOv5s pretrained weights

        # Initialize the SORT tracker
        self.tracker = Sort()

        # Object tracking data
        self.tracking_log = {}
        self.positions = {}

        # Dictionary to hold unique colors for each boat ID
        self.id_colors = {}

    def load_mask(self):
        mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask file not found: {self.mask_path}")
        return mask

    def get_color(self, obj_id):
        """
        Assign a unique color to each object ID.
        """
        if obj_id not in self.id_colors:
            # Generate a random color
            color = tuple(random.randint(0, 255) for _ in range(3))
            self.id_colors[obj_id] = color
        return self.id_colors[obj_id]

    def process_video(self):
        # Load the mask
        mask = self.load_mask()

        cap = cv2.VideoCapture(self.input_video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            self.output_video_path,
            fourcc,
            frame_rate,
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        )

        # Calculate the starting frame and ending frame
        start_frame = int(total_frames * 0.33)  # Jump to 33% of the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        max_frames = int(frame_rate * 15)  # Number of frames for 15 seconds

        frame_count = 0

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            timestamp = (start_frame + frame_count) / frame_rate  # Current time in seconds

            # Apply the mask to the frame
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

            # Run YOLOv5 detection
            results = self.model(masked_frame)
            detections = results.xyxy[0].cpu().numpy()
            boat_detections = []

            # Filter for boats (class 8 in YOLOv5)
            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection
                if int(cls) == 8:  # Ensure class 8 corresponds to boats
                    boat_detections.append([x1, y1, x2, y2, conf])

            boat_detections = np.array(boat_detections)
            if boat_detections.size == 0:
                boat_detections = np.empty((0, 5))  # Empty array with 5 columns

            # Update tracker with detections
            tracked_objects = self.tracker.update(boat_detections)

            for obj in tracked_objects:
                x1, y1, x2, y2, obj_id = obj.astype(int)

                # Record initial and last positions
                if obj_id not in self.positions:
                    self.positions[obj_id] = {'initial_x': (x1 + x2) / 2, 'last_x': (x1 + x2) / 2}
                else:
                    self.positions[obj_id]['last_x'] = (x1 + x2) / 2

                # Record timestamps
                if obj_id not in self.tracking_log:
                    self.tracking_log[obj_id] = {'in': timestamp, 'out': None}
                else:
                    self.tracking_log[obj_id]['out'] = timestamp

                # Draw bounding box
                color = self.get_color(obj_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Put label with ID
                label = f"ID: {obj_id}"
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - label_height - baseline), (x1 + label_width, y1), color, -1)  # Text background
                cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Write the frame to output video
            out.write(frame)

        cap.release()
        out.release()


    def save_to_csv(self):
        with open(self.csv_output_path, 'w', newline='') as csvfile:
            fieldnames = ['ID', 'timestamp_in', 'timestamp_out', 'duration', 'direction']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for obj_id, timestamps in self.tracking_log.items():
                if timestamps['out'] is not None:
                    duration = timestamps['out'] - timestamps['in']
                    if duration >= 10:  # Only log objects visible for 10 seconds or more
                        # Determine direction
                        delta_x = self.positions[obj_id]['last_x'] - self.positions[obj_id]['initial_x']
                        direction = "Loading" if delta_x > 0 else "Unloading"

                        writer.writerow({
                            'ID': obj_id,
                            'timestamp_in': round(timestamps['in'], 2),
                            'timestamp_out': round(timestamps['out'], 2),
                            'duration': round(duration, 2),
                            'direction': direction
                        })

        print(f"CSV log saved: {self.csv_output_path}")

    def run(self):
        print("Processing video...")
        self.process_video()
        print("Saving CSV log...")
        self.save_to_csv()
        print("Boat tracking completed.")


if __name__ == "__main__":
    input_video_path = "./data/segment_20241130_162116.mp4"
    tracker = BoatTracker(input_video_path)
    tracker.run()
