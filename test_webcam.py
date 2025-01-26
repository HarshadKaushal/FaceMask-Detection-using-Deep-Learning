import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
from datetime import datetime
import winsound  # for Windows sound
import threading  # for non-blocking sound

# Load the trained model
model = load_model('mask_detector_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize variables
mask_violations = 0
total_detections = 0
start_time = time.time()
last_alert_time = time.time()  # To control alert frequency

def play_alert():
    # Frequency = 2500Hz, Duration = 1000ms
    winsound.Beep(2500, 1000)

def alert_with_cooldown():
    global last_alert_time
    current_time = time.time()
    # Only alert if 3 seconds have passed since last alert
    if current_time - last_alert_time >= 3:
        # Use threading to prevent sound from blocking the video feed
        threading.Thread(target=play_alert, daemon=True).start()
        last_alert_time = current_time

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Create a copy for saving violations
    original_frame = frame.copy()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
    
    # Calculate FPS
    fps = 1 / (time.time() - start_time)
    start_time = time.time()
    
    # Count people in frame
    people_count = len(faces)
    
    # Flag to track if any person is without mask
    no_mask_detected = False
    
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (224, 224))
        face_roi = face_roi / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)
        
        result = model.predict(face_roi)
        
        label = "No Mask" if result[0][0] > 0.5 else "Mask"
        color = (0, 0, 255) if result[0][0] > 0.5 else (0, 255, 0)
        
        # Update violation counter and trigger alert
        if label == "No Mask":
            mask_violations += 1
            no_mask_detected = True
            
        total_detections += 1
        
        # Calculate confidence percentage
        confidence = (result[0][0] if label == "No Mask" else 1 - result[0][0]) * 100
        
        # Draw enhanced visuals
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{label} ({confidence:.1f}%)", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Trigger alert if anyone is without mask
    if no_mask_detected:
        alert_with_cooldown()
        # Add visual alert indicator
        cv2.putText(frame, "ALERT: Mask Violation!", 
                    (frame.shape[1]//2 - 100, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Add information overlay
    info_color = (255, 255, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_color, 2)
    cv2.putText(frame, f"People in frame: {people_count}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_color, 2)
    cv2.putText(frame, f"Mask Violations: {mask_violations}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_color, 2)
    
    # Add time and date
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, current_time, (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_color, 2)
    
    cv2.imshow('Mask Detection System', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()