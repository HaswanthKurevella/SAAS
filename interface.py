import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import dlib
from PIL import Image, ImageTk
from deepface import DeepFace

# Face and landmark detection setup
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1, p2):
    return ((p1.x + p2.x) // 2, (p1.y + p2.y) // 2)

def get_gaze_ratio(eye_points, landmarks, gray):
    eye_region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in eye_points], np.int32)
    height, width = gray.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [eye_region], 255)
    eye_patch = cv2.bitwise_and(gray, gray, mask=mask)
    min_x, max_x = np.min(eye_region[:, 0]), np.max(eye_region[:, 0])
    min_y, max_y = np.min(eye_region[:, 1]), np.max(eye_region[:, 1])
    gray_eye = eye_patch[min_y:max_y, min_x:max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_white = cv2.countNonZero(threshold_eye[:, 0:int(width/2)])
    right_white = cv2.countNonZero(threshold_eye[:, int(width/2):])
    return left_white / right_white if right_white != 0 else 1.0

class App:
    def __init__(self, window):
        self.window = window
        self.window.title("Student Attention Analysis")
        self.video_label = tk.Label(window)
        self.video_label.pack()
        self.status_label = tk.Label(window, text="Status: Not Started", font=("Helvetica", 14))
        self.status_label.pack()

        self.start_button = tk.Button(window, text="Start", command=self.start_camera)
        self.start_button.pack(side="left", padx=10, pady=10)

        self.stop_button = tk.Button(window, text="Stop", command=self.stop_camera)
        self.stop_button.pack(side="right", padx=10, pady=10)

        self.running = False
        self.cap = None

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image='',text='camera stopped')
        self.status_label.config(text="status:Stopped")

    def update_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        gaze_text = "Gaze: Unknown"
        emotion = "Unknown"
        status_text = "Attention: Not Interested"

        for face in faces:
            landmarks = predictor(gray, face)
            gaze_ratio_left = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, gray)
            gaze_ratio_right = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, gray)
            gaze_ratio = (gaze_ratio_left + gaze_ratio_right) / 2

            if 0.6 <= gaze_ratio <= 1.5:
                gaze_text = "Gaze: CENTER"
            elif gaze_ratio < 0.6:
                gaze_text = "Gaze: RIGHT"
            else:
                gaze_text = "Gaze: LEFT"

            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion'] if analysis else "Unknown"
            interested = gaze_text == "Gaze: CENTER" and emotion in ["happy", "surprise", "neutral"]
            status_text = "Attention: Interested" if interested else "Attention: Not Interested"

        # Update status
        self.status_label.config(text=f"{gaze_text} | Emotion: {emotion} | {status_text}")

        # Display in GUI
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.window.after(10, self.update_frame)

# Run the app
root = tk.Tk()
app = App(root)
root.mainloop()
