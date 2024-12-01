import cv2
import torch
from torchvision import models, transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from PIL import Image

# Load the image and convert to RGB
def get_count(image_path,object):
    
    image = Image.open(image_path).convert("RGB")

    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Apply the transformation
    img = transform(image)

    # Load a pre-trained Faster R-CNN model with updated weights parameter
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()

    # Perform object detection
    with torch.no_grad():
        predictions = model([img])

    # Specify the object class to count
    target_class = object  # Ensure this matches the COCO classes

    # COCO dataset classes
    COCO_CLASSES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
        'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
        'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    # Get the index of the target class
    if target_class in COCO_CLASSES:
        target_idx = COCO_CLASSES.index(target_class)
    else:
        print(f"Class '{target_class}' not found in COCO classes.")
        exit()

    # Count the number of target objects with confidence > 0.3
    count = 0
    for label, score in zip(predictions[0]['labels'], predictions[0]['scores']):
        if label.item() == target_idx and score.item() > 0.3:
            count += 1

    return (f"Number of '{target_class}' detected: {count}")