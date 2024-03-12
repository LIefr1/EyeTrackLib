import cv2
import dlib
from inspect import currentframe, getframeInfo
import math
import sys # noqa 

FIRST_FACE: int = 0


# Initialize video capture object from webcam
cap = cv2.VideoCapture("VID_4.mp4")

# Load pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(
    r"C:\Users\scher\AppData\Local\Programs\Python\Python39\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    r"C:\Users\scher\AppData\Local\Programs\Python\Python39\Lib\site-packages\cv2\data\haarcascade_eye.xml"
)

# Tracker object (will be initialized later)
tracker = None


# Initialize eye tracker (will be initialized later)
eye_tracker = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale for efficiency
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection in the first frame only
    if tracker is None:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # If a face is found, initialize the tracker
        try:
            if len(faces) > 0:
                (x, y, w, h) = faces[FIRST_FACE]
                tracker = dlib.correlation_tracker()
                rec = dlib.rectangle(x, y, x + w, y + h)

                tracker.start_track(gray, rec)
        except Exception as e:
            frameInfo = getframeInfo(currentframe())
            print(f"File:${frameInfo.filename}, line ${frameInfo.lineno +1 }: ${e}")

    # If tracker is initialized, use it to track faces
    if tracker is not None:
        tracker.update(gray)
        pos = tracker.get_position()

        try:
            # Extract bounding box coordinates and draw rectangle
            x = pos.left()
            y = pos.top()
            w = pos.right() - x
            h = pos.bottom() - y
            cv2.rectangle(
                frame,
                (math.ceil(x), math.ceil(y)),
                (math.ceil(w + x), math.ceil(h + y)),
                (255, 0, 0),
                2,
            )

            # If the eye tracker is not initialized, initialize it
            if eye_tracker is None:
                eye_tracker = dlib.correlation_tracker()
                eye_rect = dlib.rectangle(
                    int(x + w * LEFT_EYE_CENTER[0]), 
                    int(y + h * LEFT_EYE_CENTER[1]),
                    int(x + w * RIGHT_EYE_CENTER[0]),
                    int(y + h * RIGHT_EYE_CENTER[1])
                )

                eye_tracker.start_track(gray, eye_rect)

            # If the eye tracker is initialized, use it to track eyes
            if eye_tracker is not None:
                eye_tracker.update(gray)
                eye_pos = eye_tracker.get_position()

                # Extract eye coordinates and draw circles
                ex = eye_pos.left()
                ey = eye_pos.top()
                ew = eye_pos.right() - ex
                eh = eye_pos.bottom() - ey
                cv2.circle(frame, (math.ceil(ex), math.ceil(ey)), 5, (0, 255, 0), -1)
                cv2.circle(frame, (math.ceil(ex + ew), math.ceil(ey + eh)), 5, (0, 255, 0), -1)

        except Exception as e:
            frameInfo = getframeInfo(currentframe())
            print(f"File:${frameInfo.filename}, line ${frameInfo.lineno + 1}: ${e}")


    # Display resulting frame
    cv2.imshow("Face Tracker", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release capture object and close all windows
cap.release()
cv2.destroyAllWindows()
