import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils
import time
import pyttsx3
import tkinter as tk
from tkinter import simpledialog
import csv

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (3.0 * C)
    return ear

def set_alert_duration():
    global alert_duration
    alert_duration = simpledialog.askinteger("Input", "Enter the duration in seconds for the alert:")

def set_cooldown_duration():
    global cooldown_duration
    cooldown_duration = simpledialog.askinteger("Input", "Enter the cooldown duration in seconds for the alert:")

def set_alert_message():
    global text_to_speech
    text_to_speech = simpledialog.askstring("Input", "Enter the alert message:")

def save_results(frame_number, detected_blink, true_blink):
    with open('blink_detection_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([frame_number, detected_blink, true_blink])

root = tk.Tk()
root.title("Blink Detection System")

alert_button = tk.Button(root, text="Set Alert Duration", command=set_alert_duration)
alert_button.pack(pady=10)

cooldown_button = tk.Button(root, text="Set Cooldown Duration", command=set_cooldown_duration)
cooldown_button.pack(pady=10)

message_button = tk.Button(root, text="Set Alert Message", command=set_alert_message)
message_button.pack(pady=10)

start_button = tk.Button(root, text="Start Blink Detection", command=root.destroy)
start_button.pack(pady=20)

root.mainloop()

# Load Haar cascades and Dlib predictor
face_cascade = cv2.CascadeClassifier(r"D:\bhanu\desktop\projects\READ\read1\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(r"D:\bhanu\desktop\projects\READ\read1\haarcascade_mcs_eyepair_big.xml")
predictor = dlib.shape_predictor(r"D:\bhanu\desktop\projects\READ\read1\shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
engine = pyttsx3.init()

count = 0
total = 0
last_eye_detection_time = time.time()
audio_played = False
last_alert_time = 0
faces_detected = False
frame_number = 0

# Initialize results file
with open('blink_detection_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame Number', 'Detected Blink', 'True Blink'])

while True:
    success, img = cap.read()
    if not success:
        break

    frame_number += 1
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        faces_detected = True
    else:
        faces_detected = False
        count = 0
        total = 0
        audio_played = False
        last_alert_time = 0

    eyes_detected = False
    detected_blink = False

    if faces_detected:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = imgGray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                eyes_detected = True
                last_eye_detection_time = time.time()

        faces = detector(imgGray)
        for face in faces:
            landmarks = predictor(imgGray, face)
            landmarks = face_utils.shape_to_np(landmarks)
            leftEye = landmarks[42:48]
            rightEye = landmarks[36:42]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            if ear < 0.2:  # Adjusted EAR threshold
                count += 1
                detected_blink = True
            else:
                if count >= 1:
                    total += 1
                count = 0

        if not eyes_detected:
            cv2.putText(img, "Error: Eyes not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elapsed_time = int(time.time() - last_eye_detection_time)
            cv2.putText(img, "Time: {} sec".format(elapsed_time), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if elapsed_time >= alert_duration and not audio_played:
                engine.say(text_to_speech)
                engine.runAndWait()
                audio_played = True
                last_alert_time = time.time()
        else:
            audio_played = False
            count = 0

        if audio_played and time.time() - last_alert_time >= cooldown_duration:
            engine.say(text_to_speech)
            engine.runAndWait()
            last_alert_time = time.time()

    # Simulate true blink detection for the purpose of this demo
    true_blink = 'True Blink' if detected_blink else 'No Blink'

    save_results(frame_number, 'True Blink' if detected_blink else 'No Blink', true_blink)

    cv2.putText(img, "Blink Count: {}".format(total), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    watermark = "I am watching you"
    (text_width, text_height), _ = cv2.getTextSize(watermark, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_x = int(img.shape[1] / 2 - text_width / 2)
    text_y = int(img.shape[0] - text_height)
    cv2.putText(img, watermark, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
