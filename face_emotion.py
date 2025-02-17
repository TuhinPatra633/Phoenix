import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Heuristic function to classify emotions
def classify_emotion(landmarks, frame_width, frame_height):
    # Extract key landmark points
    left_eye = np.array([landmarks[145], landmarks[159]])  # Left eye top and bottom
    right_eye = np.array([landmarks[374], landmarks[386]])  # Right eye top and bottom
    mouth = np.array([landmarks[13], landmarks[14]])  # Mouth top and bottom
    brow_left = np.array([landmarks[52], landmarks[65]])  # Left eyebrow top and bottom
    brow_right = np.array([landmarks[282], landmarks[295]])  # Right eyebrow top and bottom

    # Calculate aspect ratios
    left_eye_ratio = np.linalg.norm(left_eye[0] - left_eye[1]) / frame_height
    right_eye_ratio = np.linalg.norm(right_eye[0] - right_eye[1]) / frame_height
    mouth_ratio = np.linalg.norm(mouth[0] - mouth[1]) / frame_height
    brow_left_ratio = np.linalg.norm(brow_left[0] - brow_left[1]) / frame_height
    brow_right_ratio = np.linalg.norm(brow_right[0] - brow_right[1]) / frame_height

    # Emotion classification heuristics
    if mouth_ratio > 0.04:
        if (left_eye_ratio + right_eye_ratio) > 0.03:
            return "Happy"
        return "Surprised"
    elif left_eye_ratio < 0.015 and right_eye_ratio < 0.015:
        return "Sad"
    elif brow_left_ratio > 0.015 and brow_right_ratio > 0.015:
        return "Anxious"
    else:
        return "Neutral"

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract landmarks
            landmarks = [(int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0]))
                         for pt in face_landmarks.landmark]

            # Classify emotion
            emotion = classify_emotion(landmarks, frame.shape[1], frame.shape[0])

            # Draw landmarks and emotion label
            for x, y in landmarks:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            cv2.putText(frame, f"Emotion: {emotion}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display frame
    cv2.imshow("Face Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
