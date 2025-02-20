import face_recognition as fr
import numpy as np
import cv2
import os

# Folder path containing face images
faces_folder_path = r"C:\Users\divya\Downloads\SLOW EYE\faces"

def get_face_encodings():
    # List all files in the folder
    face_names = os.listdir(faces_folder_path)
    face_encodings = []

    for i, name in enumerate(face_names):
        # Load each face image from the folder
        face = fr.load_image_file(os.path.join(faces_folder_path, name))
        # Compute face encodings
        face_encodings.append(fr.face_encodings(face)[0])

        # Extract name from the file name
        face_names[i] = os.path.splitext(name)[0]

    return face_encodings, face_names

face_encodings, face_names = get_face_encodings()

# Video capture from webcam
video = cv2.VideoCapture(0)

# Scaling factor for resizing
scl = 2

while True:
    success, image = video.read()

    # Resize the image for processing efficiency
    resized_image = cv2.resize(image, (int(image.shape[1]/scl), int(image.shape[0]/scl)))

    # Convert image to RGB format
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # Detect face locations in the image
    face_locations = fr.face_locations(rgb_image)
    unknown_encodings = fr.face_encodings(rgb_image, face_locations)

    # Compare face encodings with known encodings
    for face_encoding, face_location in zip(unknown_encodings, face_locations):
        result = fr.compare_faces(face_encodings, face_encoding, tolerance=0.4)

        if True in result:
            name = face_names[result.index(True)]
            top, right, bottom, left = face_location

            # Draw rectangle around the face
            cv2.rectangle(image, (left*scl, top*scl), (right*scl, bottom*scl), (0, 0, 255), 2)

            # Display name above the rectangle
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, name, (left*scl, bottom*scl + 20), font, 0.8, (255, 255, 255), 1)

    # Display the processed frame
    cv2.imshow("frame", image)
    k = cv2.waitKey(10) & 0xff
    if k == 27:  # Esc key to exit
        break

# Release video capture and close windows
video.release()
cv2.destroyAllWindows()
