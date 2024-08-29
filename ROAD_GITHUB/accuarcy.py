import cv2
import numpy as np
import time
import os
import tkinter as tk
from tkinter import messagebox, ttk
import csv
from sklearn.metrics import precision_score, recall_score, f1_score

class TrafficControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Control System")

        self.yellow_threshold = tk.IntVar()
        self.red_threshold = tk.IntVar()
        self.max_vehicle_limit = tk.IntVar()
        self.record_screen = tk.BooleanVar()

        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def create_widgets(self):
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(frame, text="Yellow Light Threshold:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.yellow_threshold).grid(row=0, column=1)

        ttk.Label(frame, text="Red Light Threshold:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.red_threshold).grid(row=1, column=1)

        ttk.Label(frame, text="Max Vehicle Limit:").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.max_vehicle_limit).grid(row=2, column=1)

        ttk.Checkbutton(frame, text="Record Screen", variable=self.record_screen).grid(row=3, columnspan=2)

        ttk.Button(frame, text="Start", command=self.start_processing).grid(row=4, columnspan=2)

        self.status_label = ttk.Label(frame, text="", foreground="red")
        self.status_label.grid(row=5, columnspan=2, sticky=tk.W)

    def start_processing(self):
        try:
            yellow = self.yellow_threshold.get()
            red = self.red_threshold.get()
            max_limit = self.max_vehicle_limit.get()

            if yellow < red < max_limit:
                self.status_label.config(text="Processing started...", foreground="green")
                self.process_video()
            else:
                messagebox.showerror("Invalid Input", "Ensure: Yellow < Red < Max vehicle limit.")
        except tk.TclError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for all thresholds.")

    def on_closing(self):
        self.root.quit()
        self.root.destroy()

    def read_ground_truth(self, file_path):
        ground_truth = {}
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                frame_number = int(row['frame_number'])
                count = int(row['true_count'])
                ground_truth[frame_number] = count
        return ground_truth

    def calculate_metrics(self, predictions, ground_truth):
        y_true = [ground_truth.get(i, 0) for i in range(1, len(predictions) + 1)]
        y_pred = [predictions.get(i, 0) for i in range(1, len(predictions) + 1)]

        print("y_true:", y_true)  # Debug print
        print("y_pred:", y_pred)  # Debug print

        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        return precision, recall, f1

    def process_video(self):
        # Create folders if they don't exist
        if not os.path.exists("recorded_videos"):
            os.makedirs("recorded_videos")
        if not os.path.exists("logs"):
            os.makedirs("logs")

        # Create log file and write the header if it doesn't exist
        log_file_path = os.path.join("logs", "emergency_logs.csv")
        if not os.path.exists(log_file_path):
            with open(log_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Vehicle Count"])

        # Load YOLO model
        weights_path = r'yolov4.weights'
        cfg_path = r'yolov4.cfg'
        names_path = r"coco.names"

        net = cv2.dnn.readNet(weights_path, cfg_path)
        with open(names_path, 'r') as f:
            classes = f.read().strip().split('\n')

        cap = cv2.VideoCapture(r"cars_-_1900 (720p).mp4")

        conf_threshold = 0.5
        nms_threshold = 0.4
        desired_fps = 30
        delay = int(1000 / desired_fps)

        prev_vehicle_count = 0
        paused = False
        yellow_light = False
        red_light = False
        blink_interval = 0.5
        last_blink_time = time.time()

        default_timer = 30
        current_timer = default_timer

        # Set up video writer if recording is enabled
        if self.record_screen.get():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('recorded_videos/output.avi', fourcc, desired_fps, (int(cap.get(3)), int(cap.get(4))))

        # Load ground truth data
        ground_truth = self.read_ground_truth('ground_truth.csv')
        predictions = {}

        def process_frame(frame, frame_number):
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            detections = net.forward(net.getUnconnectedOutLayersNames())

            boxes = []
            confidences = []
            for detection in detections:
                for object_detection in detection:
                    scores = object_detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > conf_threshold and class_id == 2:
                        center_x = int(object_detection[0] * frame.shape[1])
                        center_y = int(object_detection[1] * frame.shape[0])
                        w = int(object_detection[2] * frame.shape[1])
                        h = int(object_detection[3] * frame.shape[0])
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))

            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
            vehicle_count = len(indices)

            if len(indices) > 0:
                for i in indices.flatten():
                    box = boxes[i]
                    x, y, w, h = box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            predictions[frame_number] = vehicle_count
            return frame, vehicle_count

        def show_emergency_popup(vehicle_count):
            messagebox.showwarning("Emergency", "Vehicle limit exceeded! STOP!")

            # Log the vehicle count and timestamp to the CSV file
            with open(log_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), vehicle_count])

        frame_number = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if not paused:
                frame, vehicle_count = process_frame(frame, frame_number)
                cv2.putText(frame, 'Vehicle Count: {}'.format(vehicle_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)

                if vehicle_count != prev_vehicle_count:
                    print("Number of vehicles at traffic light:", vehicle_count)
                    prev_vehicle_count = vehicle_count

                if vehicle_count <= self.yellow_threshold.get():
                    yellow_light_active = True
                    red_light_active = False
                else:
                    yellow_light_active = False

                if vehicle_count >= self.red_threshold.get():
                    red_light_active = True
                    yellow_light_active = False
                else:
                    red_light_active = False

                current_time = time.time()
                if yellow_light_active and current_time - last_blink_time >= blink_interval:
                    last_blink_time = current_time
                    yellow_light = not yellow_light

                if red_light_active and current_time - last_blink_time >= blink_interval:
                    last_blink_time = current_time
                    red_light = not red_light

                if vehicle_count > self.max_vehicle_limit.get():
                    show_emergency_popup(vehicle_count)

                frame_number += 1

            if yellow_light_active and yellow_light:
                cv2.circle(frame, (frame.shape[1] - 50, 50), 15, (0, 255, 255), -1)
                cv2.putText(frame, 'YELLOW', (frame.shape[1] - 85, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            elif red_light_active and red_light:
                cv2.circle(frame, (frame.shape[1] - 50, 50), 15, (0, 0, 255), -1)
                cv2.putText(frame, 'RED', (frame.shape[1] - 70, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            elif current_timer == 0:
                yellow_light = not yellow_light
                red_light = not red_light
                current_timer = default_timer
            else:
                current_timer -= 1

            cv2.imshow('Traffic Video', frame)

            # Write the frame to the video file if recording is enabled
            if self.record_screen.get():
                out.write(frame)

            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused

        cap.release()
        if self.record_screen.get():
            out.release()
        cv2.destroyAllWindows()

        # Calculate and print accuracy metrics
        precision, recall, f1 = self.calculate_metrics(predictions, ground_truth)
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")

        self.on_closing()  # Close the tkinter window when the video processing ends


if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficControlApp(root)
