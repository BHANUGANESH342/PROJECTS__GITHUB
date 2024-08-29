import cv2

# Load pre-trained pedestrian detection model
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open video file or webcam
cap = cv2.VideoCapture(r"C:\Users\rkssp\Desktop\virtual envi\road\road\running_-_294 (720p).mp4")  # Replace 'your_video_file.mp4' with the path to your video file

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect pedestrians in the frame
    pedestrians, _ = hog.detectMultiScale(frame, winStride=(4, 4), padding=(5, 5), scale=2)

    # Draw bounding boxes around detected pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with pedestrian count
    cv2.putText(frame, 'Pedestrians: ' + str(len(pedestrians)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Pedestrian Count', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
