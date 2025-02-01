import cv2
import numpy as np
import dlib


cap=cv2.VideoCapture(0)

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def midpoint(p1,p2):
    return ((p1.x + p2.x) // 2, (p1.y + p2.y) // 2)
font = cv2.FONT_HERSHEY_PLAIN
def get_eye(eye_points,facial_landmarks):
    left_point=(facial_landmarks.part(eye_points[0]).x,facial_landmarks.part(eye_points[0]).y)
    right_point=(facial_landmarks.part(eye_points[3]).x,facial_landmarks.part(eye_points[3]).y)
    center_top=midpoint(facial_landmarks.part(eye_points[1]),facial_landmarks.part(eye_points[2]))
    center_bottom=midpoint(facial_landmarks.part(eye_points[5]),facial_landmarks.part(eye_points[4]))
    # ver_line=cv2.line(frame,center_top,center_bottom,(255,0,0),2)
    # hor_line=cv2.line(frame,left_point,right_point,(255,0,0),2)
def get_gaze_ratio(eye_points,facial_landmarks):
        eye_region=np.array([(facial_landmarks.part(eye_points[0]).x,facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x,facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x,facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x,facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x,facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x,facial_landmarks.part(eye_points[5]).y)],np.int32)
        #cv2.polylines(frame,[left_eye_region],True,(0,0,255),2)
        height,width,_=frame.shape
        mask=np.zeros((height,width),np.uint8)
        cv2.polylines(mask,[eye_region],True,255,2)
        cv2.fillPoly(mask,[eye_region],255)
        eye_patch=cv2.bitwise_and(gray,gray,mask=mask)

        min_x=np.min(eye_region[:,0])  
        max_x=np.max(eye_region[:,0])
        min_y=np.min(eye_region[:,1])
        max_y=np.max(eye_region[:,1])

        gray_eye=eye_patch[min_y:max_y,min_x:max_x]
        # gray_eye=cv2.cvtColor(eye,cv2.COLOR_BGR2GRAY)
        _,threshold_eye=cv2.threshold(gray_eye,70,255,cv2.THRESH_BINARY)
        height,width=threshold_eye.shape

        left_side_threshold=threshold_eye[0:height,0:int(width/2)]
        left_side_white=cv2.countNonZero(left_side_threshold)

        right_side_threshold=threshold_eye[0:height,int(width/2):width]
        right_side_white=cv2.countNonZero(right_side_threshold)
        gaze_ratio = 0  # Default value
        if right_side_white != 0:
            gaze_ratio = left_side_white / right_side_white
        else:
            gaze_ratio = float(9)  # You can use infinity or a large number to indicate extreme gaze to the left
        return gaze_ratio


while True:
    _,frame=cap.read()
    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    # landmarks=np.array()
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

        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41],landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47],landmarks)
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2


        cv2.putText(frame,str(gaze_ratio),(50,200),font,2,255,3)
        if gaze_ratio < 0.6:  # Looking to the RIGHT
            cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
            new_frame[:] = (0, 0, 255)  # Red for RIGHT gaze
        elif 1.6<=gaze_ratio<4.5:  # gaze_ratio > 1.1, Looking to the LEFT
            cv2.putText(frame, "LEFT", (50, 100), font, 2, (255, 0, 0), 3)
            new_frame[:] = (255, 0, 0)  # Blue for LEFT gaze
        else:
            cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 255, 0), 3)
            new_frame[:] = (0, 255, 0)  # Green for CENTER gaze
            # cv2.putText(frame,str(gaze_ratio),(50,100),font,2,(0,0,255),3)
        # # cv2.putText(frame,str(left_side_white),(50,100),font,2,(0,0,255),3)
        # # cv2.putText(frame,str(right_side_white),(50,150),font,2,(0,0,255),3)

        # threshold_eye=cv2.resize(threshold_eye,None,fx=5,fy=5)
        # eye=cv2.resize(gray_eye,None,fx=5,fy=5)
        # # cv2.imshow("Left Eye",eye)
        # cv2.imshow("Threshold Eye",threshold_eye)
        # cv2.imshow("left",left_side_threshold)
        # cv2.imshow("right",right_side_threshold)
        # # cv2.imshow("left eye thresh",left_eye_patch)


    cv2.imshow("Student Attention Analysis System",frame)
    cv2.imshow("Student Attention Analysis",new_frame)
    key=cv2.waitKey(100)
    if key==27:
        break
cap.release()
cv2.destroyAllWindows()