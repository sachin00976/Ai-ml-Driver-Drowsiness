import cv2
import time
import dlib
from scipy.spatial import distance

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)

tracker = dlib.correlation_tracker()
tracking = False
driver_face = None
eye_closed_time = 0
last_eye_open_time = time.time()
threshold = 2
drowsy = False

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)

    if not tracking or len(faces) == 0:
        if len(faces) > 0:
            largest_area = 0
            for face in faces:
                area = (face.right() - face.left()) * (face.bottom() - face.top())
                if area > largest_area:
                    largest_area = area
                    driver_face = face
            if driver_face:
                tracker.start_track(img, driver_face)
                tracking = True
    else:
        tracking_quality = tracker.update(img)
        if tracking_quality >= 8:
            pos = tracker.get_position()
            driver_face = dlib.rectangle(int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom()))
        else:
            tracking = False
            driver_face = None

    if driver_face:
        face_landmarks = dlib_facelandmark(gray, driver_face)
        leftEye = []
        rightEye = []

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))
            next_point = n+1 if n != 41 else 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(img,(x,y),(x2,y2),(0,255,0),1)

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x,y))
            next_point = n+1 if n != 47 else 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(img,(x,y),(x2,y2),(0,255,0),1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)
        EAR = (left_ear + right_ear) / 2
        EAR = round(EAR, 2)

        if EAR < 0.24:
            eye_closed_time = time.time() - last_eye_open_time
        else:
            last_eye_open_time = time.time()
            eye_closed_time = 0
            drowsy = False

        if eye_closed_time > threshold:
            drowsy = True

        if drowsy:
            cv2.putText(img,"DROWSINESS DETECTED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.putText(img,"ACTIVE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img,f"EAR: {EAR}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Driver Monitoring', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()