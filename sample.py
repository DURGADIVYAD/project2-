import cv2
import numpy as np
import dlib
from math import hypot
from twilio.rest import Client

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\divya\\Downloads\\SLOW EYE\\shape_predictor_68_face_landmarks.dat")
font = cv2.FONT_HERSHEY_PLAIN
keyboard = np.zeros((600, 1200, 3), np.uint8)
keys_set_1 = {0: "Water", 1: "Hungry", 2: "Uneasiness", 3: "Washroom", 4: "Breathing", 5: "Family Member", 6: "Emergency", 7: "Entertainment", 8: "Walking"}

def letter(letter_index, text, letter_light):
    image_path = {
        0: "C:\\Users\\divya\\Downloads\\SLOW EYE\\symbols\\0.Water.png",
        1: "C:\\Users\\divya\\Downloads\\SLOW EYE\\symbols\\1.Hungerness.png",
        2: "C:\\Users\\divya\\Downloads\\SLOW EYE\\symbols\\2.Uneasiness.png",
        3: "C:\\Users\\divya\\Downloads\\SLOW EYE\\symbols\\3.Washroom.png",
        4: "C:\\Users\\divya\\Downloads\\SLOW EYE\\symbols\\4.Breathing.png",
        5: "C:\\Users\\divya\\Downloads\\SLOW EYE\\symbols\\5.Family.png",
        6: "C:\\Users\\divya\\Downloads\\SLOW EYE\\symbols\\6.Emergency.png",
        7: "C:\\Users\\divya\\Downloads\\SLOW EYE\\symbols\\7.Entertainment.png",
        8: "C:\\Users\\divya\\Downloads\\SLOW EYE\\symbols\\8.Walking.png"
    }

    try:
        # Load the image
        image = cv2.imread(image_path[letter_index])

        # Check if the image is loaded successfully
        if image is None:
            raise Exception(f"Unable to load image {image_path[letter_index]}")

        # Resize the image
        image = cv2.resize(image, (100, 100))

        # Display the image and text together in the same box
        row = letter_index // 3  # Determine row
        col = letter_index % 3   # Determine column
        x_offset = col * 267     # Adjust x-coordinate based on column
        y_offset = row * 200     # Adjust y-coordinate based on row
        keyboard[y_offset:y_offset+100, x_offset:x_offset+100] = image
        cv2.putText(keyboard, text, (x_offset, y_offset + 120),
                    font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw a rectangle around the text
        if letter_light:
            cv2.rectangle(keyboard, (x_offset, y_offset), (x_offset+100, y_offset+100), (255, 255, 255), -1)
        else:
            cv2.rectangle(keyboard, (x_offset, y_offset), (x_offset+100, y_offset+100), (255, 0, 0), 3)

    except Exception as e:
        print(f"Error: {e}")

def midpoint(p1, p2):
    x_mid = (p1.x + p2.x) // 2
    y_mid = (p1.y + p2.y) // 2
    return x_mid, y_mid

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_length = hypot(left_point[0] - right_point[0], left_point[1] - right_point[1])
    ver_line_length = hypot(center_top[0] - center_bottom[0], center_top[1] - center_bottom[1])

    ratio = hor_line_length / ver_line_length
    return ratio

def get_gaze_ratio(eye_points, facial_landmarks, frame, gray):
    # Get coordinates of eye landmarks
    left_eye_region = np.array([
        (facial_landmarks[eye_points[0]][0], facial_landmarks[eye_points[0]][1]),
        (facial_landmarks[eye_points[1]][0], facial_landmarks[eye_points[1]][1]),
        (facial_landmarks[eye_points[2]][0], facial_landmarks[eye_points[2]][1]),
        (facial_landmarks[eye_points[3]][0], facial_landmarks[eye_points[3]][1]),
        (facial_landmarks[eye_points[4]][0], facial_landmarks[eye_points[4]][1]),
        (facial_landmarks[eye_points[5]][0], facial_landmarks[eye_points[5]][1])
    ], np.int32)

    # Get dimensions of the frame
    height, width, _ = frame.shape

    # Create a mask for the left eye region
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)

    # Apply the mask to the grayscale image to isolate the left eye
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    # Calculate threshold for binarization
    _, threshold_eye = cv2.threshold(eye, 70, 255, cv2.THRESH_BINARY)

    # Calculate gaze ratio
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[:, 0:width // 2]
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_threshold = threshold_eye[:, width // 2:width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white

    return gaze_ratio


cap = cv2.VideoCapture(0)

board = np.zeros((500, 500,), np.uint8)
board[:] = 255

frames = 0
blinking_frames = 0
letter_index = 0
text = ""
message_sent = False  # Flag to track if a message has been sent

# Delay in seconds between updates of the keyboard display
delay_between_updates = 0.2  # Adjust as needed

while True:
    _, frame = cap.read()
    frames += 1
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    keyboard[:] = (0, 0, 0)

    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    active_letter = keys_set_1[letter_index]

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > 5.7:
            cv2.putText(frame, "BLINKING", (50, 150), font, 4, (255, 0, 0), thickness=3)
            blinking_frames += 1
            frames -= 1

            if blinking_frames == 2:
                text += active_letter
        else:
            blinking_frames = 0

    if frames == 30:
        letter_index += 1
        frames = 0
    if letter_index == 9:
        letter_index = 0

    for i in range(9):
        if i == letter_index:
            light = True
        else:
            light = False
        letter(i, keys_set_1[i], light)

    cv2.putText(board, text, (10, 100), font, 4, 0, 3)
    cv2.imshow("Board", board)
    cv2.imshow("Frame", frame)
    cv2.imshow("Keyboard", keyboard)

    if text in keys_set_1.values() and not message_sent:
        account_sid = 'AC1917ab183cb7d9629700f793275f4d26'
        auth_token = 'd909f2be2b78980eceb32e4f8497a22c'

        client = Client(account_sid,auth_token)

        if text == "Water":
            message = client.messages.create(
                body="give me water",
                from_="+16812282043",
                to="+918951553796"
            )
        elif text == "Hungry":
            message = client.messages.create(
                body="give me food",
                from_="+16812282043",
                to="+918951553796"
            )
        elif text == "Uneasiness":
            message = client.messages.create(
                body="come here feeling uneasiness",
                from_="+16812282043",
                to="+918951553796"
            )
        elif text == "Washroom":
            message = client.messages.create(
                body="want to go to washroom",
                from_="+16812282043",
                to="+918951553796"
            )
        elif text == "Breathing":
            message = client.messages.create(
                body="i am getting breathing problem",
                from_="+16812282043",
                to="+918951553796"
            )
        elif text == "Family Member":
            message = client.messages.create(
                body="call my family member",
                from_="+16812282043",
                to="+918951553796"
            )
        elif text == "Emergency":
            message = client.messages.create(
                body="It's an emergency, please come soon",
                from_="+16812282043",
                to="+918951553796"
            )
        elif text == "Entertainment":
            message = client.messages.create(
                body="I'm bored, can I watch TV",
                from_="+16812282043",
                to="+918951553796"
            )
        elif text == "Walking":
            message = client.messages.create(
                body="I would like to take a stroll outside",
                from_="+16812282043",
                to="+918951553796"
            )

        print(message.sid)
        message_sent = True  # Set message_sent to True after sending message

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
