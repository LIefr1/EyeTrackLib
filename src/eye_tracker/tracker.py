import cv2
import torch
import numpy as np
from landmark_detector.model import Network


class Tracker:
    def __init__(
        self,
        frame,
        path: str = "models/resnet152_eye_only.pth",
        model=Network(model_name="resnet152"),
    ):
        self.model = model
        self.path = path
        self.frame = frame
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

    def _get_face(self):
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        return face_cascade.detectMultiScale(self.gray, 1.1, 4)

    def _get_landmarks(self):
        try:
            self.model.load_state_dict(torch.load(self.path))
        except Exception as e:
            (e)

    def update(self, frame):
        pass

    def get_position(self):
        pass
