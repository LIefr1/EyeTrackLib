import math as m
import cv2 as cv
import dlib
import inspect
import pyautogui as pg
from utils import print_camera_properties, move_mouse
# from LK_Optical_flow import LK


def tracker_init(cascade, gray_frame):
    track_object = cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    if len(track_object) > 0:
        try:
            (x, y, w, h) = track_object[0]
            tracker = dlib.correlation_tracker()
            rec = dlib.rectangle(x, y, x + w, y + h)
            tracker.start_track(gray_frame, rec)
            return tracker
        except Exception as e:
            frameInfo = inspect.getframeInfo(inspect.currentframe())
            print(f"File:${frameInfo.filename}, line ${frameInfo.lineno + 1}: ${e}")


def get_roi(x: int, y: int, w: int, h: int, frame):
    return frame[y : y + h, x : x + w]


cap = cv.VideoCapture(0)
cap.set(propId=cv.CAP_PROP_FRAME_WIDTH, value=2560)
cap.set(propId=cv.CAP_PROP_FRAME_HEIGHT, value=1080)
if not cap:
    print("Cannot open camera")
    exit()
print_camera_properties()


# if cap.isOpened():
#     sleep(5)

feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
)

face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_alt.xml"
)
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")
face_tracker, eye_tracker = None, None


_, old_frame = cap.read()

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)


while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    if face_tracker is None:
        face_tracker = tracker_init(face_cascade, gray)
        # print("set up face tracker")
    if face_tracker is not None:
        face_tracker.update(gray)
        pos = face_tracker.get_position()
        try:
            x = pos.left()
            y = pos.top()
            w = pos.right() - x
            h = pos.bottom() - y
            face_roi = get_roi(m.floor(y), m.floor(x), m.floor(w), m.floor(h), gray)

            if previous_gray is not None and previous_pts is not None:
                next_pts, status, err = cv.calcOpticalFlowPyrLK(
                    previous_gray,
                    gray,
                    previous_pts,
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
                )

                # Update previous_gray and previous_pts for next frame
                previous_gray = gray.copy()
                previous_pts = next_pts[status == 1].tolist()

            else:
                # Initialize previous_gray and previous_pts for the first frame
                previous_gray = gray.copy()
                mask = np.zeros_like(gray)
                mask[y : y + h, x : x + w] = 255
                previous_pts = cv.goodFeaturesToTrack(
                    gray, mask=mask, maxCorners=100, qualityLevel=0.3, minDistance=10
                ).tolist()

            # Draw tracking points (optional)
            for i, pt in enumerate(previous_pts):
                cv.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)

            screen_width, screen_height = pg.size()
            invert_x = screen_width - m.ceil(x * 1.5)
            invert_y = screen_height - m.ceil(y * 1.2)
            # move_mouse(invert_x, m.ceil(y * 1.5))
            # LK(cap)
            if eye_tracker is None:
                eye_tracker = tracker_init(eye_cascade, gray)
                if not eye_tracker:
                    print("set up eye tracker")
            cv.rectangle(
                frame,
                (m.ceil(x), m.ceil(y)),
                (m.ceil(w + x), m.ceil(h + y)),
                (255, 0, 0),
                2,
            )
            # move_mouse(m.ceil(y), m.ceil(y))
        except Exception as e:
            frameInfo = inspect.getframeinfo(inspect.currentframe())
            print(f"File:${frameInfo.filename}, line ${frameInfo.lineno + 1}: ${e}")

    if eye_tracker is not None:
        eye_tracker.update(gray)
        pos = eye_tracker.get_position()
        try:
            x = pos.left()
            y = pos.top()
            w = pos.right() - x
            h = pos.bottom() - y
            print(x, y, w, h)
            invert_x = screen_width - m.ceil(x * 1.5)
            move_mouse(invert_x, m.ceil(y * 1.5))
            eye_roi = get_roi(m.floor(x), m.floor(y), m.floor(w), m.floor(h), gray)
            cv.rectangle(
                frame,
                (m.ceil(x), m.ceil(y)),
                (m.ceil(w + x), m.ceil(h + y)),
                (255, 0, 0),
                2,
            )
        except Exception as e:
            frameInfo = inspect.getframeInfo(inspect.currentframe())
            print(f"File:${frameInfo.filename}, line ${frameInfo.lineno + 1}: ${e}")
    cv.imshow("Face Tracker", frame)

    # Exit loop on 'q' key pres s
    if cv.waitKey(30) & 0xFF == 27:
        break
