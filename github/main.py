import cv2
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import os
import pyqrcode
from PIL import Image, ImageTk
from datetime import datetime
import pyttsx3
import webbrowser
import numpy as np

class QRCodeApp:
    def __init__(self, master):
        self.master = master
        self.master.title("QR Code Action")

        tk.Label(master, text="What would you like to do?").pack()

        tk.Button(master, text="Generate QR Code", command=self.generate_qr_code).pack(pady=10)
        tk.Button(master, text="Scan QR Code (Camera)", command=self.scan_qr_code_camera).pack(pady=10)
        tk.Button(master, text="Scan Generated QR Code", command=self.scan_generated_qr_code).pack(pady=10)
        tk.Button(master, text="Detect Face (Capture & Convert)", command=self.detect_face_and_convert).pack(pady=10)

        # Bind the Escape key to exit the main loop
        master.bind("<Escape>", self.exit_main_loop)

        # Add this line to create a folder for cropped QR codes
        self.crop_folder = "output/crop"
        os.makedirs(self.crop_folder, exist_ok=True)

        # Add this line to create an output folder for scanned QR codes
        self.output_folder = "output/selection"
        os.makedirs(self.output_folder, exist_ok=True)

        # Add this line to create a folder for scanned links
        self.links_folder = "output/links"
        os.makedirs(self.links_folder, exist_ok=True)

    def exit_main_loop(self, event):
        self.master.destroy()

    def generate_qr_code(self):
        try:
            qr_type, user_input = self.get_user_input()

            today_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_file_path = os.path.join(self.output_folder, f"{qr_type}_{today_date}.png")

            self.create_qr(user_input, output_file_path)
            print(f"QR code ({qr_type}) saved to: {output_file_path}")

            self.show_congratulations_popup()

            engine = pyttsx3.init()
            engine.say("Thank you! Your QR code is generated.")
            engine.runAndWait()

            # Ask if the user wants to scan the generated QR code
            answer = messagebox.askyesno("Scan Generated QR Code", "Do you want to scan the generated QR code?")
            if answer:
                # Scan the generated QR code
                self.scan_generated_qr_code(output_file_path)

        except Exception as e:
            print(f"Error generating QR code: {e}")

    def scan_qr_code_camera(self):
        try:
            cap = cv2.VideoCapture(0)

            while True:
                ret, frame = cap.read()

                detector = cv2.QRCodeDetector()
                retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(frame)

                if retval:
                    for value in decoded_info:
                        print(f"Decoded QR code: {value}")

                        # Save scanned QR codes in a date-wise text file
                        today_date = datetime.now().strftime("%Y-%m-%d")
                        output_file_path = os.path.join(self.links_folder, f"scanned_links_{today_date}.txt")

                        with open(output_file_path, "a") as file:
                            file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {value}\n")

                        self.process_scanned_data(value, None, points)

                        if self.is_link(value):
                            self.master.destroy()
                            return

                cv2.imshow('QR Code Scanner', frame)

                # Check for the 'Esc' key to exit
                if cv2.waitKey(1) == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error scanning QR code through the camera: {e}")

    def is_link(self, data):
        return data.startswith("http") or data.startswith("www")

    def scan_generated_qr_code(self, file_path=None):
        try:
            if file_path is None:
                file_path = filedialog.askopenfilename(title="Select QR Code Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

                if not file_path:
                    print("No file selected. Exiting.")
                    return

            img = cv2.imread(file_path)

            # Process the image
            detector = cv2.QRCodeDetector()
            retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(img)

            if retval:
                for value in decoded_info:
                    print(f"Decoded QR code: {value}")

                    self.process_scanned_data(value, file_path, points)

                    self.show_decoded_text_popup(value)

                    break

            # Display the image
            cv2.imshow('Scanned QR Code', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error scanning QR code through the camera: {e}")

    def process_scanned_data(self, data, file_path=None, pts=None):
        print(f"Scanned QR code data: {data}")
        if self.is_link(data):
            engine = pyttsx3.init()
            engine.say("Scanning successful. Opening the link.")
            engine.runAndWait()

            self.save_and_show_cropped_qr(pts)

            webbrowser.open(data)

        else:
            #print("Scanned data does not match the generated QR code.")
            pass

    def save_and_show_cropped_qr(self, pts):
        try:
            img = cv2.imread(file_path)
            rect_pts = np.array(pts, dtype=int)
            rect_pts = rect_pts.reshape((-1, 1, 2))

            mask = np.zeros_like(img)

            # Fill the mask with the contour of the QR code
            cv2.drawContours(mask, [rect_pts], -1, (255, 255, 255), thickness=cv2.FILLED)

            # Bitwise-AND operation to get the cropped QR code
            cropped_qr = cv2.bitwise_and(img, mask)

            today_date = datetime.now().strftime("%Y-%m-%d")
            output_file_path = os.path.join(self.crop_folder, f"cropped_qr_{today_date}.png")
            cv2.imwrite(output_file_path, cropped_qr)
            print(f"Cropped QR code saved to: {output_file_path}")

            cv2.imshow('Cropped QR Code', cropped_qr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            decoded_text = self.decode_qr_code(output_file_path)
            self.show_decoded_text_popup(decoded_text)

            cv2.imshow('Cropped Image', cropped_qr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error saving and showing cropped QR code: {e}")

    def decode_qr_code(self, image_path):
        image = cv2.imread(image_path)
        detector = cv2.QRCodeDetector()
        retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(image)
        decoded_text = decoded_info[0] if retval else "Decoding failed"
        return decoded_text

    def show_decoded_text_popup(self, decoded_text):
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Decoded Text", f"The decoded text is:\n\n{decoded_text}")

    def get_user_input(self):
        print("Choose the type of QR code:")
        print("1. Text")
        print("2. Video Link")
        print("3. Payment Information")
        print("4. Contact Information")
        print("5. Event Details")
        print("6. Face Detection and Conversion")

        choice = int(simpledialog.askstring("QR Code Type", "Enter the number corresponding to your choice:"))

        if choice == 1:
            return "Text", simpledialog.askstring("Enter Text", "Enter the text to encode:")
        elif choice == 2:
            return "Video Link", simpledialog.askstring("Enter Video Link", "Enter the video link to encode:")
        elif choice == 3:
            return "Payment Information", simpledialog.askstring("Enter Payment Information", "Enter the payment information to encode:")
        elif choice == 4:
            return "Contact Information", self.get_contact_information()
        elif choice == 5:
            return "Event Details", self.get_event_details()
        elif choice == 6:
            return "Face Detection", "Face Detection"
        else:
            print("Invalid choice. Exiting.")
            exit()

    def get_contact_information(self):
        name = simpledialog.askstring("Contact Information", "Enter the contact name:")
        number = simpledialog.askstring("Contact Information", "Enter the contact number:")

        contact_info = f"BEGIN:VCARD\nVERSION:3.0\nFN:{name}\nTEL:{number}\nEND:VCARD"
        return contact_info

    def get_event_details(self):
        event_name = simpledialog.askstring("Event Details", "Enter the name of the event:")
        additional_text = simpledialog.askstring("Event Details", "Enter additional text for the event details:")

        event_info = f"BEGIN:VEVENT\nSUMMARY:{event_name}\nDTSTART:20220101T120000\nDTEND:20220101T140000\nLOCATION:Example Venue\nDESCRIPTION:{additional_text}\nEND:VEVENT"
        return event_info

    def create_qr(self, data, output_path):
        qr = pyqrcode.QRCode(data)
        qr.png(output_path, scale=10)

    def show_congratulations_popup(self):
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Congratulations!", "Your QR code is generated.\nThank you!")

    def detect_face_and_convert(self):
        try:
            cap = cv2.VideoCapture(0)
            face_cascade_path = r"haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(face_cascade_path)

            while True:
                ret, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around the face
                cv2.imshow("Live Camera", frame)

                key = cv2.waitKey(1)
                if key == ord('s'):
                    image_name = self.ask_user_for_image_name()
                    if image_name:
                        if len(faces) > 0:
                            x, y, w, h = faces[0]
                            face_roi = frame[y:y + h, x:x + w]
                            image_path = f"output/{image_name}.jpg"
                            cv2.imwrite(image_path, face_roi)
                            response = messagebox.askquestion("Convert to QR Code",
                                                              "Do you want to convert this image to a QR code?")
                            if response == 'yes':
                                qr_code_path = self.convert_to_qr_code(image_path, image_name)
                                if qr_code_path:
                                    response = messagebox.askquestion("Display QR Code",
                                                                      "Do you want to display the generated QR code?")
                                    if response == 'yes':
                                        self.display_qr_code(qr_code_path)
                                        # Destroy the window after displaying the QR code
                                        cv2.destroyAllWindows()
                                    messagebox.showinfo("QR Code Generated",
                                                        f"QR code generated successfully and saved at: {qr_code_path}")

                        else:
                            messagebox.showerror("No Face Detected", "No face detected in the frame.")

                elif key == 27:  # ESC key to exit
                    break

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error detecting face and converting: {e}")

    def ask_user_for_image_name(self):
        image_name = simpledialog.askstring("Image Name", "Enter the name for the captured image:")
        return image_name

    def convert_to_qr_code(self, image_path, image_name):
        qr_code_path = f"output/{image_name}_qr_code.png"
        qr = pyqrcode.create(image_path)
        qr.png(qr_code_path, scale=8)
        return qr_code_path

    def display_qr_code(self, qr_code_path):
        qr_image = Image.open(qr_code_path)
        qr_image.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = QRCodeApp(root)
    root.mainloop()
