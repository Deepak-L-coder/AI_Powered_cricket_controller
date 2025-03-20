import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import threading

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Start capturing video
cap = cv2.VideoCapture(0)

# Variables for tracking bat and hand movement
prev_wrist_y, prev_wrist_x = None, None
swing_threshold_y = 35  # Reduce threshold for faster response
swing_threshold_x = 15
time_delay = 0.15  # Faster response time
last_swing_time = 0  # For debounce mechanism
cooldown = 0.5  # 500ms cooldown

# Define HSV color range for detecting the bat (adjust these values based on bat color)
lower_color = np.array([10, 100, 100])   # Example: Yellow/Orange bat
upper_color = np.array([30, 255, 255])  

# Function to detect the bat
def detect_bat(hsv_frame, frame):
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)
    
    # Reduce noise using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    


    
    bat_detected = False
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 800:  # Increased threshold for more accuracy
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Draw a box around the bat
            bat_detected = True

    return bat_detected, mask

# Mouse click function to avoid lag
def click_mouse():
    pyautogui.mouseDown()
    time.sleep(0.05)
    pyautogui.mouseUp()

# Main loop for video processing
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Convert to different color spaces
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect Bat
    bat_detected, bat_mask = detect_bat(hsv, frame)

    # Process pose detection
    result = pose.process(frame_rgb)
    
    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark
        
        # Track right wrist movement
        wrist_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame.shape[1])
        wrist_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame.shape[0])

        if prev_wrist_y is not None and prev_wrist_x is not None:
            y_movement = prev_wrist_y - wrist_y
            x_movement = abs(prev_wrist_x - wrist_x)

            # Detect a swing if bat is detected + fast wrist movement
            if bat_detected and y_movement > swing_threshold_y and x_movement > swing_threshold_x:
                # Debounce mechanism to prevent multiple detections
                if time.time() - last_swing_time > cooldown:
                    print("Swing detected! Tapping mouse...")
                    
                    # Run mouse click in a separate thread to prevent lag
                    threading.Thread(target=click_mouse).start()

                    last_swing_time = time.time()  # Update last swing time

        # Apply a low-pass filter to smooth tracking (reduces jitter)
        prev_wrist_y = (prev_wrist_y * 0.7 + wrist_y * 0.3) if prev_wrist_y else wrist_y
        prev_wrist_x = (prev_wrist_x * 0.7 + wrist_x * 0.3) if prev_wrist_x else wrist_x

    # Show the video feed
    cv2.imshow("Bat Swing Detection", frame)
    cv2.imshow("Bat Mask", bat_mask)  # Show the detected bat mask

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()