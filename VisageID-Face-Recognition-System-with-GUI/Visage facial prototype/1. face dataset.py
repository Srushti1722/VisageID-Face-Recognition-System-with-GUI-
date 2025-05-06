import cv2
import os

# Initialize camera
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW for better performance on Windows
cam.set(3, 640)  # Set width
cam.set(4, 480)  # Set height

if not cam.isOpened():
    print("[ERROR] Camera could not be opened!")
    exit()

# Load face detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Validate user ID input
while True:
    face_id = input("\nEnter a numeric user ID: ")
    if face_id.isdigit():
        break
    print("Invalid input. Please enter a numeric value.")

print("\n[INFO] Initializing face capture. Look at the camera...")

# Create dataset folder if not exists
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Capture face samples
count = 0
while True:
    ret, img = cam.read()
    if not ret:
        print("[ERROR] Failed to capture image.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        img_name = f"{dataset_path}/User.{face_id}.{count}.jpg"
        cv2.imwrite(img_name, gray[y:y + h, x:x + w])

        cv2.putText(img, f"Captured: {count}/30", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Face Capture', img)

    # Break conditions
    if cv2.waitKey(1) & 0xFF == 27 or count >= 30:  # ESC key to exit
        break

print("\n[INFO] Face capture complete. Cleaning up...")
cam.release()
cv2.destroyAllWindows()
