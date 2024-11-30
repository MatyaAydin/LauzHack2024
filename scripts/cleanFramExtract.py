import cv2

# Load the video
video_path = "./data/live_feed_5min2.mp4"
cap = cv2.VideoCapture(video_path)

# Go to the last frame
cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)

# Read the last frame
ret, frame = cap.read()
if ret:
    # Save the last frame as an image
    cv2.imwrite("./data/clean_frame.jpg", frame)
    print("Last frame saved as 'clean_frame.jpg'")
else:
    print("Failed to extract the last frame")

cap.release()
