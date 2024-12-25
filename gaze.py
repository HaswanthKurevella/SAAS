import cv2
import numpy as np
import dlib


cap=cv2.VideoCapture(0)

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def midpoint(p1,p2):
    return ((p1.x + p2.x) // 2, (p1.y + p2.y) // 2)
def get_eye(eye_points,facial_landmarks):
    left_point=(facial_landmarks.part(eye_points[0]).x,facial_landmarks.part(eye_points[0]).y)
    right_point=(facial_landmarks.part(eye_points[3]).x,facial_landmarks.part(eye_points[3]).y)
    center_top=midpoint(facial_landmarks.part(eye_points[1]),facial_landmarks.part(eye_points[2]))
    center_bottom=midpoint(facial_landmarks.part(eye_points[5]),facial_landmarks.part(eye_points[4]))
    # ver_line=cv2.line(frame,center_top,center_bottom,(255,0,0),2)
    # hor_line=cv2.line(frame,left_point,right_point,(255,0,0),2)
        

while True:
    _,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    for face in faces:

# eye detection

        x,y=face.left(),face.top()
        x1,y1=face.right(),face.bottom()
        # cv2.rectangle(frame,(x,y),(x1,y1),(255,0,0),2)  #color of the rectangle
        landmarks=predictor(gray,face)
        left_eye=get_eye([36,37,38,39,40,41],landmarks)  # 36,37,38,39,40,41 for left eye 
        right_eye=get_eye([42,43,44,45,46,47],landmarks)  # 42,43,44,45,46,47 for right eye
        

        # x=landmarks.part(36).x
        # y=landmarks.part(36).y
        # cv2.circle(frame,(x,y),3,(255,0,0),2)
        # print(face)


#Blinking detection (sleeping or not?)

#gaze Detection
        left_eye_region=np.array([(landmarks.part(36).x,landmarks.part(36).y),
                                  (landmarks.part(37).x,landmarks.part(37).y),
                                  (landmarks.part(38).x,landmarks.part(38).y),
                                  (landmarks.part(39).x,landmarks.part(39).y),
                                  (landmarks.part(40).x,landmarks.part(40).y),
                                  (landmarks.part(41).x,landmarks.part(41).y)],np.int32)
        #cv2.polylines(frame,[left_eye_region],True,(0,0,255),2)
        height,width,_=frame.shape
        mask=np.zeros((height,width),np.uint8)
        cv2.polylines(mask,[left_eye_region],True,255,2)
        cv2.fillPoly(mask,[left_eye_region],255)
        left_eye_patch=cv2.bitwise_and(gray,gray,mask=mask)

        min_x=np.min(left_eye_region[:,0])  
        max_x=np.max(left_eye_region[:,0])
        min_y=np.min(left_eye_region[:,1])
        max_y=np.max(left_eye_region[:,1])

        gray_eye=left_eye_patch[min_y:max_y,min_x:max_x]
        # gray_eye=cv2.cvtColor(eye,cv2.COLOR_BGR2GRAY)
        _,threshold_eye=cv2.threshold(gray_eye,70,255,cv2.THRESH_BINARY)
        height,width=threshold_eye.shape

        left_side_threshold=threshold_eye[0:height,0:int(width/2)]
        left_side_white=cv2.countNonZero(left_side_threshold)

        right_side_threshold=threshold_eye[0:height,int(width/2):width]
        right_side_white=cv2.countNonZero(right_side_threshold)
        cv2.putText(frame,str(left_side_white),(),font,2,(0,0,255),3)

        threshold_eye=cv2.resize(threshold_eye,None,fx=5,fy=5)
        eye=cv2.resize(gray_eye,None,fx=5,fy=5)
        # cv2.imshow("Left Eye",eye)
        cv2.imshow("Threshold Eye",threshold_eye)
        cv2.imshow("left",left_side_threshold)
        cv2.imshow("right",right_side_threshold)
        # cv2.imshow("left eye thresh",left_eye_patch)


    cv2.imshow("Student Attention Analysis System",frame)
    key=cv2.waitKey(1)
    if key==27:
        break
cap.release()
cv2.destroyAllWindows()