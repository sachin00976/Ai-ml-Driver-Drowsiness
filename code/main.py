import cv2
import time
import dlib
from scipy.spatial import distance
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)

facepresent=0
counter=0
drowssy=0


while cap.isOpened():
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

           

    faces = hog_face_detector(gray)
    if(len(faces)>0):
        counter=0
    if(len(faces)>0 and facepresent==0):
        last_eye_open_time = time.time()
        facepresent=1
    else :
        counter+=1
        if(counter>=10):
            facepresent=0
            cv2.putText(img,"No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36,42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(img,(x,y),(x2,y2),(0,255,0),1)

        for n in range(42,48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x,y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(img,(x,y),(x2,y2),(0,255,0),1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        EAR = (left_ear+right_ear)/2
        EAR = round(EAR,2)
        

        eyes_closed = True
        any_eye_open = False
        if(EAR>0.24):
            any_eye_open=True

        if not any_eye_open:
            
            drowssy=1;

        else:
            drowssy=0;
       
        if(drowssy):
            cv2.putText(img,"Drowsiness Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
           
        else :
            cv2.putText(img,"Active",(20,50),
            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4)
           
            cv2.putText(img,"EAR:",(480,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4)
            cv2.putText(img,str(EAR),(550,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4)
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
