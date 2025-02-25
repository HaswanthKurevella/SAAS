import cv2
import numpy as np
import dlib
import os
from deepface import DeepFace
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
# import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Helper function for gaze detection
def get_gaze_ratio(eye_points, facial_landmarks, gray):
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

# Path to dataset folders
dataset_path = "Student-engagement-dataset"  # Change this to your dataset path
categories = {
    "Engaged": 1,  # Engaged = 1
    "Not Engaged": 0  # Not Engaged = 0
}

true_labels = []
gaze_predictions = []
emotion_predictions = []
integrated_predictions = []

# Process each image from subfolders
for category, label in categories.items():
    category_path = os.path.join(dataset_path, category)

    for subfolder in os.listdir(category_path):  # Iterate through subfolders
        subfolder_path = os.path.join(category_path, subfolder)

        for img_name in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            gaze_result, emotion_result, integrated_result = 0, 0, 0  # Default Not Engaged

            for face in faces:
                landmarks = predictor(gray, face)
                gaze_ratio_left = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, gray)
                gaze_ratio_right = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, gray)
                gaze_ratio = (gaze_ratio_left + gaze_ratio_right) / 2

                # Gaze classification
                if 0.9 <= gaze_ratio <= 1.2:
                    gaze_result = 1  # Engaged
                else:
                    gaze_result = 0  # Not Engaged

                # Emotion classification using DeepFace
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                if analysis:
                    emotion = analysis[0]['dominant_emotion']
                    if emotion in ["happy", "surprise", "neutral"]:
                        emotion_result = 1  # Engaged
                    else:
                        emotion_result = 0  # Not Engaged

                # Integrated classification
                integrated_result = 1 if (gaze_result == 1 and emotion_result == 1) else 0

            # Append results
            true_labels.append(label)
            gaze_predictions.append(gaze_result)
            emotion_predictions.append(emotion_result)
            integrated_predictions.append(integrated_result)

# Calculate accuracy
gaze_accuracy = accuracy_score(true_labels, gaze_predictions)
emotion_accuracy = accuracy_score(true_labels, emotion_predictions)
integrated_accuracy = accuracy_score(true_labels, integrated_predictions)

# Plot results
labels = ['Gaze Detection', 'Emotion Recognition', 'Integrated System']
accuracies = [gaze_accuracy, emotion_accuracy, integrated_accuracy]

plt.figure(figsize=(8, 5))
plt.bar(labels, accuracies, color=['blue', 'green', 'red'])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Comparison of Engagement Detection Methods")
plt.show()
