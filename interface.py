import tkinter as tk
from tkinter import Label, Button
import cv2
import numpy as np
import dlib
from PIL import Image, ImageTk
from deepface import DeepFace
import threading

class AttentionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Attention Detector")
        
        self.status_label = Label(root, text="Status: Idle", font=("Arial", 14))
        self.status_label.pack(pady=10)

        self.video_label = Label(root)
        self.video_label.pack()


        self.start_button = Button(root, text="Start", command=self.start_camera)
        self.start_button.pack(side=tk.LEFT, padx=20, pady=10)

        self.stop_button = Button(root, text="Stop", command=self.stop_camera)
        self.stop_button.pack(side=tk.RIGHT, padx=20, pady=10)

        self.cap = None
        self.running = False
        self.frame_count = 0
        self.current_emotion = "Analyzing..."

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def midpoint(self, p1, p2):
        return ((p1.x + p2.x) // 2, (p1.y + p2.y) // 2)

    def get_gaze_ratio(self, eye_points, facial_landmarks, gray):
        eye_region = np.array([(facial_landmarks.part(point).x, facial_landmarks.part(point).y) for point in eye_points], np.int32)
        height, width = gray.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [eye_region], 255)
        eye_patch = cv2.bitwise_and(gray, gray, mask=mask)
        min_x, max_x = np.min(eye_region[:, 0]), np.max(eye_region[:, 0])
        min_y, max_y = np.min(eye_region[:, 1]), np.max(eye_region[:, 1])
        gray_eye = eye_patch[min_y:max_y, min_x:max_x]
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
        height, width = threshold_eye.shape
        left_side_white = cv2.countNonZero(threshold_eye[:, 0:int(width/2)])
        right_side_white = cv2.countNonZero(threshold_eye[:, int(width/2):width])
        gaze_ratio = left_side_white / right_side_white if right_side_white != 0 else 1.0
        return gaze_ratio

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.frame_count = 0
        self.current_emotion = "Analyzing..."
        self.update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.config(image='', text='Camera stopped')
        self.status_label.config(text="Status: Stopped")

    def update_frame(self):
        if not self.running or not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        gaze_text = "Gaze: Unknown"
        interested = False

        for face in faces:
            landmarks = self.predictor(gray, face)
            gaze_ratio_left = self.get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, gray)
            gaze_ratio_right = self.get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, gray)
            gaze_ratio = (gaze_ratio_left + gaze_ratio_right) / 2

            if 0.6 <= gaze_ratio <= 1.2:
                gaze_text = "Gaze: CENTER"
            elif gaze_ratio < 0.6:
                gaze_text = "Gaze: RIGHT"
            else:
                gaze_text = "Gaze: LEFT"

            # Run DeepFace every 10 frames in a separate thread
            if self.frame_count % 10 == 0:
                threading.Thread(target=self.update_emotion, args=(frame.copy(),)).start()

            # Check interest based on last known emotion
            interested = gaze_text == "Gaze: CENTER" and self.current_emotion in ["happy", "surprise", "neutral"]

        status_text = "Attention: Interested" if interested else "Attention: Not Interested"
        self.status_label.config(text=f"{gaze_text} | Emotion: {self.current_emotion} | {status_text}")

        # Display the video frame
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def update_emotion(self, frame):
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            self.current_emotion = analysis[0]['dominant_emotion']
        except Exception as e:
            self.current_emotion = "Unknown"

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = AttentionApp(root)
    root.mainloop()

