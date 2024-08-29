import numpy as np
import cv2
import pickle
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# Set parameters
frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.5  # PROBABILITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX

# Load the trained model
with open("model_trained1.p", "rb") as pickle_in:
    model = pickle.load(pickle_in)

# Define preprocessing function
def preprocessing(img):
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255
    return img

# Define function to get class name from class number
def getClassName(classNo):
    classes = {
        0: 'Speed Limit 20 km/h', 1: 'Speed Limit 30 km/h', 2: 'Speed Limit 50 km/h',
        3: 'Speed Limit 60 km/h', 4: 'Speed Limit 70 km/h', 5: 'Speed Limit 80 km/h',
        6: 'End of Speed Limit 80 km/h', 7: 'Speed Limit 100 km/h', 8: 'Speed Limit 120 km/h',
        9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
        11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield',
        14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited',
        17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left',
        20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road',
        23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work',
        26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
        29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
        32: 'End of all speed and passing limits', 33: 'Turn right ahead',
        34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
        37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
        40: 'Roundabout mandatory', 41: 'End of no passing',
        42: 'End of no passing by vehicles over 3.5 metric tons'
    }
    return classes.get(classNo, 'Unknown')

# Define function for detecting signs from images in a folder
def detect_from_folder(folder_path):
    output_folder = "output_images"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            imgOrignal = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_ANYCOLOR)
            if imgOrignal is not None:
                img = preprocessing(imgOrignal)
                img_input = img.reshape(1, 32, 32, 1)
                predictions = model.predict(img_input)
                classIndex = np.argmax(predictions)
                probabilityValue = np.amax(predictions)
                if probabilityValue > threshold:
                    className = getClassName(classIndex)
                    cv2.putText(imgOrignal, f"{className}: {probabilityValue*100:.2f}%", (20, 50), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Detected Image', imgOrignal)
                    cv2.waitKey(0)
            else:
                print("Error: Unable to load image", filename)

# Define function for live webcam detection
def detect_from_webcam():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = preprocessing(frame)
        img_input = img.reshape(1, 32, 32, 1)
        predictions = model.predict(img_input)
        classIndex = np.argmax(predictions)
        probabilityValue = np.amax(predictions)

        if probabilityValue > threshold:
            className = getClassName(classIndex)
            cv2.putText(frame, f"{className}: {probabilityValue*100:.2f}%", (20, 50), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Traffic Sign Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Define GUI function
def create_gui():
    root = tk.Tk()
    root.title("Traffic Sign Detector")

    def detect_from_folder_gui():
        folder_path = filedialog.askdirectory(title="Select folder with images")
        if not folder_path:
            return
        detect_from_folder(folder_path)

    def detect_from_webcam_gui():
        detect_from_webcam()

    folder_button = tk.Button(root, text="Detect from Folder", command=detect_from_folder_gui)
    folder_button.pack()

    webcam_button = tk.Button(root, text="Detect from Webcam", command=detect_from_webcam_gui)
    webcam_button.pack()

    root.mainloop()

if __name__ == "__main__":
    create_gui()
