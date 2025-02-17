import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize Mediapipe Hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

cap = cv2.VideoCapture(0)
last_right_click_time = 0  # To track the last right-click action time
right_click_cooldown = 1  # Cooldown time in seconds

# Variables to track wrist movement for scrolling
previous_wrist_y = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror the frame for natural cursor movement
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark coordinates
            landmarks = hand_landmarks.landmark
            index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            wrist = landmarks[mp_hands.HandLandmark.WRIST]

            # Convert normalized coordinates to screen coordinates
            x = int(index_finger_tip.x * screen_width)
            y = int(index_finger_tip.y * screen_height)

            # Gesture detection
            index_thumb_distance = ((index_finger_tip.x - thumb_tip.x) ** 2 + (index_finger_tip.y - thumb_tip.y) ** 2) ** 0.5
            index_middle_distance = ((index_finger_tip.x - middle_finger_tip.x) ** 2 + (index_finger_tip.y - middle_finger_tip.y) ** 2) ** 0.5

            # Wrist up and down for scrolling
            current_wrist_y = wrist.y
            if previous_wrist_y is not None:
                if current_wrist_y < previous_wrist_y - 0.02:  # Wrist moved up
                    pyautogui.scroll(2)  # Scroll up
                    cv2.putText(frame, 'Scrolling Up', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    print('Scrolling Up')
                elif current_wrist_y > previous_wrist_y + 0.02:  # Wrist moved down
                    pyautogui.scroll(-2)  # Scroll down
                    cv2.putText(frame, 'Scrolling Down', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    print('Scrolling Down')
            previous_wrist_y = current_wrist_y

            # Pinch gesture for click
            if index_thumb_distance < 0.05:
                pyautogui.leftClick(x, y)
                cv2.putText(frame, 'Pinch Click', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print('Left Click')

            # Right click with index + middle finger (if cooldown allows)
            elif index_middle_distance < 0.05:
                current_time = time.time()
                if current_time - last_right_click_time > right_click_cooldown:
                    pyautogui.rightClick(x, y)
                    last_right_click_time = current_time
                    cv2.putText(frame, 'Right Click', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    print('Right Click')

            # Single index finger for cursor movement
            elif index_middle_distance > 0.2 and index_thumb_distance > 0.2:
                pyautogui.moveTo(x, y)
                cv2.putText(frame, 'Moving Cursor', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print('Moving Cursor')
           

    # Show the video feed
    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()