import cv2 as cv
import numpy as np
import dlib
import ctypes as ct
import screeninfo as si
import sys


def move_mouse_ct(x: int, y: int):
    print("moving mouse to: ", x, y)
    ct.windll.user32.SetCursorPos(x, y)
    # click_if_stable_params(x=x, y=y)


# Initialize dlib's face detector (HOG-based) and the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Function to extract the eye coordinates from facial landmarks
def get_eye_points(shape):
    # Eye landmark points according to dlib's 68-point model
    left_eye_pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(36, 42)])
    right_eye_pts = np.array(
        [[shape.part(i).x, shape.part(i).y] for i in range(42, 48)]
    )
    return left_eye_pts, right_eye_pts


def get_eye_boxes(landmarks) -> dict:
    left_eye = [
        landmarks[36],
        landmarks[37],
        landmarks[38],
        landmarks[39],
        landmarks[40],
        landmarks[41],
    ]
    right_eye = [
        landmarks[42],
        landmarks[43],
        landmarks[44],
        landmarks[45],
        landmarks[46],
        landmarks[47],
    ]

    left_eye_x = [point[0] for point in left_eye]
    left_eye_y = [point[1] for point in left_eye]

    right_eye_x = [point[0] for point in right_eye]
    right_eye_y = [point[1] for point in right_eye]

    return {
        "left_eye": {
            "top_left": (int(min(left_eye_x)), int(min(left_eye_y))),
            "bottom_right": (int(max(left_eye_x)), int(max(left_eye_y))),
        },
        "right_eye": {
            "top_left": (int(min(right_eye_x)), int(min(right_eye_y))),
            "bottom_right": (int(max(right_eye_x)), int(max(right_eye_y))),
        },
    }


cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
)

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

# Detect faces
faces = detector(old_gray)
if len(faces) > 0:
    # Assume only the first detected face is of interest
    shape = predictor(old_gray, faces[0])
    left_eye_pts, right_eye_pts = get_eye_points(shape)
    eye_pts = np.concatenate((left_eye_pts, right_eye_pts), axis=0).astype(np.float32)
    p0 = eye_pts.reshape(-1, 1, 2)
else:
    p0 = np.empty((0, 1, 2))

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

frame_count = 0

while True:
    ret, frame = cap.read()
    frame_count += 1
    print("Frame count: ", frame_count)
    if not ret:
        print("No frames grabbed!")
        break
    np.set_printoptions(threshold=np.inf)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    WIDTH = si.get_monitors()[0].width

    # calculate optical flow
    if len(p0) > 0:
        p1, st, err = cv.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()  # Flatten the array to a 1-dimensional array
                c, d = old.ravel()
                mask = cv.line(
                    mask,
                    (int(a), int(b)),
                    (int(c), int(d)),
                    color[i % len(color)].tolist(),
                    2,
                )
                boxes = get_eye_boxes(shape)
                print(boxes)

                # move_mouse_ct(x=int(WIDTH - (a * 4.0)), y=int((b * 1.6875)))
                frame = cv.circle(
                    frame, (int(a), int(b)), 5, color[i % len(color)].tolist(), -1
                )

            # img = cv.add(frame, mask)

            cv.imshow("frame", frame)

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        else:
            # Re-detect the facial landmarks if no good points are found
            faces = detector(frame_gray)
            if len(faces) > 0:
                shape = predictor(frame_gray, faces[0])
                left_eye_pts, right_eye_pts = get_eye_points(shape)
                eye_pts = np.concatenate((left_eye_pts, right_eye_pts), axis=0).astype(
                    np.float32
                )
                p0 = eye_pts.reshape(-1, 1, 2)
            else:
                p0 = np.empty((0, 1, 2))
    if frame_count % 3 == 0:
        faces = detector(frame_gray)
        if len(faces) > 0:
            shape = predictor(frame_gray, faces[0])
            left_eye_pts, right_eye_pts = get_eye_points(shape)
            eye_pts = np.concatenate((left_eye_pts, right_eye_pts), axis=0).astype(
                np.float32
            )
            p0 = eye_pts.reshape(-1, 1, 2)
        else:
            p0 = np.empty((0, 1, 2))

    k = cv.waitKey(30) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
cap.release()
