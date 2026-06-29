import cv2
import os
import numpy as np
from pygame import mixer
import tensorflow as tf
import time # Added for better control
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D

# 1. Initialize Sound
mixer.init()
try:
    sound = mixer.Sound('alarm.wav')
except:
    print("Alarm sound not found.")

# 2. MANUALLY DEFINE THE MODEL ARCHITECTURE
def build_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
        MaxPooling2D(pool_size=(1,1)),
        Conv2D(32,(3,3),activation='relu'),
        MaxPooling2D(pool_size=(1,1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(1,1)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    return model

# 3. Create and Load Weights
model = build_model()
model_path = os.path.join("models", "cnnCat2.h5")
model.load_weights(model_path)
print("✅ System Ready. Press 'q' in the camera window to exit.")

# 4. Standard Setup
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
cap = cv2.VideoCapture(0)
score = 0

while(True):
    ret, frame = cap.read()
    if not ret: break
    
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, minNeighbors=1, scaleFactor=1.1)

    # If no eyes are detected, we still show the frame
    if len(eyes) == 0:
        cv2.putText(frame, "No Eyes Detected", (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)

    for (x,y,w,h) in eyes:
        eye = gray[y:y+h, x:x+w]
        eye = cv2.resize(eye, (24,24))
        eye = eye / 255.0
        eye = eye.reshape(24, 24, 1)
        eye = np.expand_dims(eye, axis=0)
        
        prediction = model.predict(eye, verbose=0)
        
        if prediction[0][0] > 0.30: # Closed
            score += 1
            cv2.putText(frame, "Closed", (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            if score > 20:
                try: sound.play()
                except: pass
        else: # Open
            score = max(0, score - 1)
            cv2.putText(frame, "Open", (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
        cv2.putText(frame, f'Score: {score}', (120, height-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)

    cv2.imshow('Drowsiness Detection', frame)
    
    # --- THE FIX FOR STOPPING ---
    # Increased to 30ms to give the OS time to catch the 'q' key
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
# Final force-close for Windows
for i in range(1, 5):
    cv2.waitKey(1)