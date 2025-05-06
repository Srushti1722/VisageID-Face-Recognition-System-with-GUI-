import cv2
import numpy as np
import os
import threading
import sqlite3
import customtkinter as ctk
from PIL import Image, ImageTk
from plyer import notification

def initialize_db():
    conn = sqlite3.connect("user_data.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

initialize_db()

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("800x600")
        
        self.running = False  # Control flag for stopping operations
        
        # Create main frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill=ctk.BOTH, expand=True)
        
        # Status label
        self.status_label = ctk.CTkLabel(self.main_frame, text="Status: Idle", font=("Arial", 16))
        self.status_label.pack(pady=10)
        
        # Name entry
        self.name_entry = ctk.CTkEntry(self.main_frame, placeholder_text="Enter Name")
        self.name_entry.pack(pady=10)
        
        # Buttons frame
        self.buttons_frame = ctk.CTkFrame(self.main_frame)
        self.buttons_frame.pack(pady=20)
        
        # Capture Image Button
        self.capture_btn = ctk.CTkButton(
            self.buttons_frame, text="Capture Image", command=self.start_capture_thread
        )
        self.capture_btn.grid(row=0, column=0, padx=10)
        
        # Train Model Button
        self.train_btn = ctk.CTkButton(
            self.buttons_frame, text="Train Model", command=self.start_train_thread
        )
        self.train_btn.grid(row=0, column=1, padx=10)
        
        # Recognize Face Button
        self.recognize_btn = ctk.CTkButton(
            self.buttons_frame, text="Recognize Face", command=self.start_recognition_thread
        )
        self.recognize_btn.grid(row=0, column=2, padx=10)

        # Cancel Button
        self.cancel_btn = ctk.CTkButton(
            self.buttons_frame,
            text="Cancel ⏹",
            command=self.cancel_operation,
            fg_color="orange",
            hover_color="darkorange"
        )
        self.cancel_btn.grid(row=1, column=0, padx=10, pady=10)

        # Quit Button
        self.quit_btn = ctk.CTkButton(
            self.buttons_frame,
            text="Quit ❌",
            command=self.quit_app,
            fg_color="red",
            hover_color="darkred"
        )
        self.quit_btn.grid(row=1, column=1, padx=10, pady=10)
    
    def update_status(self, message):
        self.status_label.configure(text=f"Status: {message}")
        print(f"DEBUG: {message}")  # Debugging

    def cancel_operation(self):
        """Stops the current operation."""
        self.running = False
        self.update_status("Operation Canceled")
        print("DEBUG: Operation canceled by user.")

    def start_capture_thread(self):
        if not self.running:
            threading.Thread(target=self.capture_image, daemon=True).start()
    
    def start_train_thread(self):
        threading.Thread(target=self.train_model, daemon=True).start()
    
    def start_recognition_thread(self):
        if not self.running:
            threading.Thread(target=self.recognize_face, daemon=True).start()

    def capture_image(self):
        self.running = True
        self.update_status("Capturing Images...")
        name = self.name_entry.get().strip()
        if not name:
            self.update_status("Enter a valid name!")
            return
        
        conn = sqlite3.connect("user_data.db")
        c = conn.cursor()
        c.execute("INSERT INTO users (name) VALUES (?)", (name,))
        user_id = c.lastrowid
        conn.commit()
        conn.close()
        
        print(f"DEBUG: Capturing images for User ID {user_id}, Name: {name}")
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.update_status("ERROR: Could not open camera!")
            return

        sample_num = 0
        os.makedirs("dataset", exist_ok=True)
        
        while self.running and sample_num < 50:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                sample_num += 1
                cv2.imwrite(f"dataset/{user_id}_{sample_num}.jpg", gray[y:y+h, x:x+w])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            cv2.imshow("Capturing Face", frame)
            if sample_num >= 50 or cv2.waitKey(1) & 0xFF == ord('q') or not self.running:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.update_status("Image Capture Completed")
        self.running = False

    def train_model(self):
        self.update_status("Training model...")
        recognizer = cv2.face.LBPHFaceRecognizer.create(radius=1, neighbors=8, grid_x=8, grid_y=8)

        path = "dataset"
        if not os.path.exists(path) or not os.listdir(path):
            self.update_status("No images found for training!")
            return
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        faces, ids = [], []
        for image_path in image_paths:
            try:
                img = Image.open(image_path).convert('L')
                img = img.resize((100, 100))  # Reduce resolution to speed up training
                img_np = np.array(img, 'uint8')
                id_ = int(os.path.basename(image_path).split("_")[0])
                faces.append(img_np)
                ids.append(id_)
            except Exception as e:
                print(f"DEBUG: Error processing {image_path} - {e}")

        if not faces:
            self.update_status("No valid images found for training!")
            return

        recognizer.train(faces, np.array(ids))
        recognizer.save("trainer.yml")

        self.update_status("Training Completed ✅ (Optimized)")
        print("DEBUG: Model trained successfully with optimized parameters.")

        

    def recognize_face(self):
        self.running = True
        self.update_status("Recognition mode active...")
        
        if not os.path.exists("trainer.yml"):
            self.update_status("No trained model found!")
            return
        
        recognizer = cv2.face.LBPHFaceRecognizer.create()

        recognizer.read("trainer.yml")
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cap = cv2.VideoCapture(0)

        conn = sqlite3.connect("user_data.db")
        c = conn.cursor()
        c.execute("SELECT id, name FROM users")
        users = dict(c.fetchall())  # Mapping ID to name
        conn.close()
        
        if not cap.isOpened():
            self.update_status("ERROR: Could not open camera!")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                name = users.get(id_, "Unknown")
                cv2.putText(frame, f"{name} ({confidence:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            cv2.imshow("Recognizing Face", frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or not self.running:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.update_status("Face Recognition Completed")

    def quit_app(self):
        self.running = False
        cv2.destroyAllWindows()
        self.root.quit()
        self.root.destroy()
    def send_notification(self, title, message):
    
        notification.notify(
            title=title,
            message=message,
            app_name="Face Recognition System",
            timeout=5  # Notification disappears after 5 seconds
            )

if __name__ == "__main__":
    root = ctk.CTk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.quit_app)
    root.mainloop()
