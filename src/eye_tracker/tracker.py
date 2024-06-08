from src.eye_tracker.landmark_predictor import Predictor

import cv2 as cv
import numpy as np
import sys


class Tracker:
    def __init__(
        self,
        feature_params=dict(maxCorners=50, qualityLevel=0.3, minDistance=7, blockSize=7),
        lk_params=dict(
            winSize=(244, 244),
            maxLevel=2,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
        ),
    ):
        self.color = np.random.randint(0, 255, (100, 3))
        self.predictor = Predictor()
        self.feature_params = feature_params
        self.lk_params = lk_params

    @staticmethod
    def get_faces(gray):
        face_cascade = cv.CascadeClassifier(
            cv.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        return face_cascade.detectMultiScale(gray, 1.1, 4)

    def detect_landmarks(
        self,
        gray,
        face,
    ):
        if face is not None:
            shape = self.predictor.predict(gray, face)
            left_eye_pts, right_eye_pts = self.get_eye_points(shape[0])

            eye_pts = np.concatenate((left_eye_pts, right_eye_pts), axis=0).astype(np.float32)
            print("eye points: ", eye_pts)
            return eye_pts.reshape(-1, 1, 2)

        else:
            return np.empty((0, 1, 2))

    @staticmethod
    def get_eye_points(shape):
        print(shape)

        left_eye_pts = np.array([shape[i] for i in range(36, 42)])

        right_eye_pts = np.array([shape[i] for i in range(42, 48)])

        return left_eye_pts, right_eye_pts

    @staticmethod
    def get_eye_boxes(landmarks, gray) -> dict:
        left_eye_points, right_eye_points = landmarks
        left_eye_np = np.array(left_eye_points, dtype=np.int32)
        right_eye_np = np.array(right_eye_points, dtype=np.int32)

        # Create bounding boxes around the eyes
        (x, y, w, h) = cv.boundingRect(left_eye_np)
        left_eye_region = gray[y : y + h, x : x + w]
        (x, y, w, h) = cv.boundingRect(right_eye_np)
        right_eye_region = gray[y : y + h, x : x + w]

        return left_eye_region, right_eye_region

    def calculate_LK(
        self,
        previos_gray,
        frame_gray,
        p0,
    ):
        p1, st, err = cv.calcOpticalFlowPyrLK(previos_gray, frame_gray, p0, None, **self.lk_params)
        if p1 is not None:
            good_old = p0[st == 1]
            good_new = p1[st == 1]

            return (good_old, good_new)
        else:
            raise ValueError("No optical flow found")
        pass

    def draw(self, frame, points, mask) -> None:
        new, old = points

        for i, (new, old) in enumerate(zip(new, old)):
            x_new, y_new = new.ravel()  # flatten the array to a 1-dimensional array

            x_old, y_old = old.ravel()

            mask = cv.line(
                mask,
                (int(x_new), int(y_new)),
                (int(x_old), int(y_old)),
                self.color[i].tolist(),
                2,
            )

            cv.circle(frame, (int(x_new), int(y_new)), 5, self.color[i].tolist(), -1)

            return cv.add(frame, mask)
