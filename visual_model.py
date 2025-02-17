import cv2
import torch
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from gtts import gTTS
from ultralytics import YOLO
import torchvision.transforms as transforms

# Function to convert text to speech
def speak(text):
    timestamp = int(time.time())  # Unique identifier
    audio_file = f"output_{timestamp}.mp3"
    tts = gTTS(text=text, lang='en')
    tts.save(audio_file)
    os.system(f"mpg321 {audio_file}")  # Use a suitable player for your OS

# Load YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")

# Load MiDaS depth estimation model
depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
depth_model.eval()
depth_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Function to determine object position more accurately
def get_position(bbox, frame_width):
    x_center = (bbox[0] + bbox[2]) / 2
    if x_center < frame_width / 4:
        return "left"
    elif x_center > 3 * frame_width / 4:
        return "right"
    else:
        return "center"

# Track previously detected objects to avoid redundant announcements
previous_detections = {}

# Open camera
cap = cv2.VideoCapture(0)
frame_count = 0
max_frames = 50  # Increase max frames for better analysis
capture_interval = 3  # 3 second between frames

time.sleep(2)  # Warm-up time for camera
while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    time.sleep(capture_interval)  # Enforce processing interval
    frame_height, frame_width, _ = frame.shape

    # Run YOLOv8 segmentation inference
    results = model(frame)

    # Convert frame to tensor and estimate depth
    input_tensor = depth_transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0)
    with torch.no_grad():
        depth_map = depth_model(input_tensor).squeeze().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())  # Normalize
    
    new_detections = {}
    detections = []

    for result in results:
        if result.masks is not None:
            for segment, box in zip(result.masks.data, result.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls[0])]
                conf = box.conf[0]
                object_depth = np.mean(depth_map[y1:y2, x1:x2])
                position = get_position((x1, y1, x2, y2), frame_width)
                
                obj_key = f"{label}_{position}"  # Unique identifier per object
                new_detections[obj_key] = object_depth
                
                if obj_key not in previous_detections or abs(previous_detections[obj_key] - object_depth) > 1.1:
                    detections.append(f"{label} at {object_depth:.2f} meters on the {position}")
                
                # Draw bounding box and segmentation mask
                mask = segment.cpu().numpy().astype(np.uint8) * 255
                mask_color = np.zeros_like(frame, dtype=np.uint8)
                mask_color[mask > 0] = (0, 255, 0)  # Green mask
                frame = cv2.addWeighted(frame, 1.0, mask_color, 0.5, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f} Depth: {object_depth:.2f} Pos: {position}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Announce only new detections
    if detections:
        announcement = "Detected: " + ", ".join(detections)
        speak(announcement)
    
    previous_detections = new_detections  # Update stored detections
    
    # Display the image
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.pause(0.01)
    plt.clf()
    
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
