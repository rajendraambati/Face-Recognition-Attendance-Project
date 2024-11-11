import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path for images and attendance file
path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)

# Load images and names
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:  # Check if the image was loaded properly
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:  # Ensure encoding is found
            encodeList.append(encode[0])
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'a') as f:  # Append mode to add new entries
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        f.write(f'{name},{dtString}\n')

# Find encodings for known faces
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Convert the frame to RGB (face_recognition uses RGB)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces and their encodings in the current frame
    facesCurFrame = face_recognition.face_locations(imgRGB)
    encodesCurFrame = face_recognition.face_encodings(imgRGB, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)

            # Draw rectangle around the recognized face
            top_left = (faceLoc[3], faceLoc[0])
            bottom_right = (faceLoc[1], faceLoc[2])
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(img, name, (faceLoc[3] + 6, faceLoc[0] - 6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Mark attendance for recognized faces
            markAttendance(name)

    # Display the resulting frame with bounding boxes
    cv2.imshow('Webcam', img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()