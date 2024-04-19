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
