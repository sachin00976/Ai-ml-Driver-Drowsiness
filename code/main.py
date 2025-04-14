import cv2
import time
import dlib
from scipy.spatial import distance
from playsound import playsound
from pygame import mixer

mixer.init()
mixer.music.load("beep_warning.mp3") 
mixer.music.set_volume(0.7)

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def enhance_eye_detection(gray, eye_region):
    if eye_region[1] < 0 or eye_region[3] > gray.shape[0] or eye_region[0] < 0 or eye_region[2] > gray.shape[1]:
        return gray
    eye_roi = gray[eye_region[1]:eye_region[3], eye_region[0]:eye_region[2]]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(eye_roi)
    enhanced_eye = gray.copy()
    enhanced_eye[eye_region[1]:eye_region[3], eye_region[0]:eye_region[2]] = enhanced
    return enhanced_eye

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)

tracker = dlib.correlation_tracker()
tracking = False
driver_face = None
eye_closed_time = 0
last_eye_open_time = time.time()
threshold = .3
drowsy = False
show_question = False

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

        left_eye_region = (face_landmarks.part(36).x, face_landmarks.part(37).y, 
                           face_landmarks.part(39).x, face_landmarks.part(41).y)
        right_eye_region = (face_landmarks.part(42).x, face_landmarks.part(43).y, 
                            face_landmarks.part(45).x, face_landmarks.part(47).y)

        gray = enhance_eye_detection(gray, left_eye_region)
        gray = enhance_eye_detection(gray, right_eye_region)

        for n in range(36, 42):
            x, y = face_landmarks.part(n).x, face_landmarks.part(n).y
            leftEye.append((x, y))

        for n in range(42, 48):
            x, y = face_landmarks.part(n).x, face_landmarks.part(n).y
            rightEye.append((x, y))

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
            show_question = False

        if eye_closed_time > threshold:
            drowsy = True
            show_question = True

        if drowsy:
            cv2.putText(img,"DROWSINESS DETECTED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            if not mixer.music.get_busy():
                mixer.music.play()
            if show_question:
                cv2.putText(img,"2 + 3 = ?", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        else:
            cv2.putText(img,"ACTIVE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img,f"EAR: {EAR}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Driver Monitoring', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('5'):
        show_question = False

cap.release()
cv2.destroyAllWindows()
