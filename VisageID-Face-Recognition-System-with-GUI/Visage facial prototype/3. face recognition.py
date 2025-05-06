import cv2
import numpy as np
import os

# Paths
TRAINER_PATH = 'trainer/trainer.yml'
CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Load face recognizer and cascade classifier
recognizer = cv2.face.LBPHFaceRecognizer_create()

if not os.path.exists(TRAINER_PATH):
    print("\n[ERROR] Trainer file not found! Please train the model first.")
    exit()

recognizer.read(TRAINER_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX

# Names (IDs should match training dataset)
names = ['Unknown', 'Vikash', 'Jack']  # Ensure IDs start at index 1

# Initialize webcam
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, 640)  # Width
cam.set(4, 480)  # Height

# Define minimum face size for recognition
minW = int(0.1 * cam.get(3))
minH = int(0.1 * cam.get(4))

print("\n[INFO] Face Recognition Running... Press 'ESC' to exit.")

while True:
    ret, img = cam.read()
    if not ret:
        print("\n[ERROR] Camera not detected.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(minW, minH)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if confidence < 100:  # "0" is perfect match
            name = names[face_id] if face_id < len(names) else "Unknown"
            conf_text = f"{round(100 - confidence)}%"
            color = (0, 255, 0) if confidence < 50 else (0, 0, 255)  # Green for high, Red for low confidence
        else:
            name = "Unknown"
            conf_text = f"{round(100 - confidence)}%"
            color = (0, 0, 255)

        # Display name and confidence
        cv2.putText(img, str(name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, conf_text, (x + 5, y + h - 5), font, 1, color, 2)

    cv2.imshow('Face Recognition', img)

    k = cv2.waitKey(10) & 0xFF
    if k == 27:  # ESC key
        break

# Cleanup
print("\n[INFO] Exiting program and releasing resources.")
cam.release()
cv2.destroyAllWindows()
