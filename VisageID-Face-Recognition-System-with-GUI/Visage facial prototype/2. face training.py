import cv2
import numpy as np
import os

# Paths
DATASET_PATH = 'dataset'
TRAINER_PATH = 'trainer/trainer.yml'
CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# Create trainer directory if not exists
if not os.path.exists("trainer"):
    os.makedirs("trainer")

# Initialize face recognizer and detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(CASCADE_PATH)

# Function to fetch images and label data
def get_images_and_labels(dataset_path):
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg') or f.endswith('.png')]
    
    if not image_paths:
        print("\n[ERROR] No images found in dataset. Please capture images first.")
        return [], []

    face_samples = []
    ids = []

    for i, image_path in enumerate(image_paths, start=1):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARNING] Skipping invalid image: {image_path}")
            continue

        face_id = os.path.basename(image_path).split(".")[1]
        if not face_id.isdigit():
            print(f"[WARNING] Skipping file with invalid ID format: {image_path}")
            continue

        face_id = int(face_id)
        faces = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_samples.append(img[y:y + h, x:x + w])
            ids.append(face_id)

        print(f"[INFO] Processed {i}/{len(image_paths)} images.", end="\r")

    return face_samples, ids

print("\n[INFO] Training faces. This may take a few seconds...")
faces, ids = get_images_and_labels(DATASET_PATH)

if faces:
    recognizer.train(faces, np.array(ids))
    recognizer.write(TRAINER_PATH)
    print(f"\n[INFO] Training complete. {len(np.unique(ids))} unique faces trained.")
else:
    print("\n[ERROR] No valid face samples found. Training aborted.")
