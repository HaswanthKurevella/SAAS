from deepface import DeepFace
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']
        
        cv2.putText(frame, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if emotion in ["happy", "surprise", "neutral"]:
            status = "Interested"
            color = (0, 255, 0)  # Green
        else:
            status = "Not Interested"
            color = (0, 0, 255)  # Red
        
        cv2.putText(frame, status, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    except Exception as e:
        print("Error:", e)

    cv2.imshow("Interest Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
