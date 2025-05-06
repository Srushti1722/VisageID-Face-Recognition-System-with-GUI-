import tkinter as tk
from tkinter import messagebox
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import time
from database import insertUserInfo, selectUserInfo

window = tk.Tk()
window.title("Face Recognition System")
window.geometry('1280x720')
window.configure(background='white')

def clear_name():
    txt_name.delete(0, 'end')
    message.configure(text="")

def TakeImages():
    name = txt_name.get().strip()
    if not name.isalpha():
        message.configure(text="Enter a valid name!")
        return

    Id = str(int(time.time()))  # Unique ID
    dataset_path = "dataset"
    os.makedirs(dataset_path, exist_ok=True)

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    sampleNum = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.2, 6, minSize=(100, 100))
        
        for (x, y, w, h) in faces:
            sampleNum += 1
            cv2.imwrite(f"{dataset_path}/{name}.{Id}.{sampleNum}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow('Capturing Images', img)
        
        if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()
    insertUserInfo(Id, name)
    message.configure(text=f"Images Saved for ID: {Id}, Name: {name}")

def TrainImages():
    path = 'dataset'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces, Ids = getImagesAndLabels(path)
    
    if len(faces) > 0:
        recognizer.train(faces, np.array(Ids))
        recognizer.save("trainer/trainer.yml")
        message.configure(text="Training Completed!")
    else:
        message.configure(text="No images found for training!")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces, Ids = [], []
    
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    
    return faces, Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer/trainer.yml")
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    df = pd.read_csv("UserInfo/UserInfo.csv")
    
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 6, minSize=(100, 100))
        
        for (x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 85:
                name = df[df['Id'] == Id]['Name'].values[0]
                cv2.putText(im, f"{name} ({conf:.2f})", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                cv2.putText(im, "Unknown", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(im, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        cv2.imshow('Face Recognition', im)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

frame = tk.LabelFrame(window, padx=50, pady=50)
frame.grid(row=3, column=0, columnspan=10)

txt_name = tk.Entry(window, width=30)
txt_name.grid(row=1, column=1)
message = tk.Label(window, text="", width=30)
message.grid(row=1, column=4)

takeImg = tk.Button(frame, text="Take Images", command=TakeImages, width=20, height=2)
takeImg.grid(row=6, column=1, pady=10)

trainImg = tk.Button(frame, text="Train Images", command=TrainImages, width=20, height=2)
trainImg.grid(row=6, column=2, pady=10)

trackImg = tk.Button(frame, text="Track Images", command=TrackImages, width=20, height=2)
trackImg.grid(row=6, column=3, pady=10)

quitButton = tk.Button(frame, text="Quit", command=window.destroy, width=20, height=2)
quitButton.grid(row=7, column=2, pady=10)

window.mainloop()