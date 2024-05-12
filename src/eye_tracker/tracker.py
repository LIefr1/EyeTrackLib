import cv2
import torch
import logging
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

from landmark_detector.model import Network


class Tracker:
    def __init__(
        self,
        prev_frame,
        path: str = "models/resnet152_eye_only.pth",
        model=Network(model_name="resnet152"),
    ):
        self.path = path
        self.prev_frame = prev_frame
        self.prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)

        self.color = np.random.randint(0, 255, (100, 3))
        self.feature_params = dict(
            maxCorners=50, qualityLevel=0.3, minDistance=7, blockSize=7
        )
        self.lk_params = dict(
            winSize=(244, 244),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        self.mask = np.zeros_like(self.prev_frame)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model, path)
        self.model.eval()

    def get_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        return face_cascade.detectMultiScale(gray, 1.1, 4)

    def _load_model(self, model, path):
        try:
            model.load_state_dict(torch.load(path, map_location=self.device))
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(e)
        finally:
            return model

    def _preprocess(self, x, y, w, h, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = gray[y : y + h, x : x + w]
        image = TF.resize(Image.fromarray(image), size=(224, 224))
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])
        return image.unsqueeze(0)

    def get_landmarks(self, x, y, w, h, frame):
        all_landmarks = []
        with torch.no_grad():
            landmarks = self.model(self._preprocess(x, y, w, h, frame))
        landmarks = (landmarks.view(68, 2).cpu().detach().numpy() + 0.5) * np.array(
            [[w, h]]
        ) + np.array([[x, y]])
        all_landmarks.append(landmarks)
        return all_landmarks

    def get_eye_boxes(self, frame, landmarks) -> dict:
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

    def _get_initial_points(self):
        faces = self.get_faces(self.prev_frame)
        all_boxes = []
        for x, y, w, h in faces:
            all_landmarks = self.get_landmarks(x, y, w, h, self.prev_frame)
            for landmarks in all_landmarks:
                eye_boxes = self.get_eye_boxes(self.prev_frame, landmarks)
                all_boxes.append(eye_boxes)
        gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        all_points = []
        for box in all_boxes:
            cropped_gray = gray[
                box["left_eye"]["top_left"][1] : box["left_eye"]["bottom_right"][1],
                box["left_eye"]["top_left"][0] : box["left_eye"]["bottom_right"][0],
            ]
            all_points.append(
                cv2.goodFeaturesToTrack(cropped_gray, mask=None, **self.feature_params)
            )
        return all_points

    def optical_flow(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        all_points = []
        initial_points = self._get_initial_points()
        for p0 in initial_points:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, p0, None, **self.lk_params
            )
            if p1 is not None:
                tracked_points = p1[st == 1]
                prev_points = p0[st == 1]
                all_points.append({"new": tracked_points, "old": prev_points})
        return all_points

    def get_position(self):
        raise NotImplementedError

    def draw(self, frame):
        faces = self.get_faces(frame)
        for x, y, w, h in faces:
            all_landmarks = self.get_landmarks(x, y, w, h, frame)
            all_points = self.optical_flow(frame)
            for idx, landmarks in enumerate(all_landmarks):
                for i, (new, old) in enumerate(
                    zip(all_points[idx]["new"], all_points[idx]["old"])
                ):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(
                        self.mask,
                        (int(a), int(b)),
                        (int(c), int(d)),
                        self.color[i].tolist(),
                        2,
                    )
                    cv2.circle(frame, (int(a), int(b)), 5, self.color[i].tolist(), -1)
                eye_boxes = self.get_eye_boxes(frame, landmarks)
                cv2.rectangle(
                    frame,
                    eye_boxes["left_eye"]["top_left"],
                    eye_boxes["left_eye"]["bottom_right"],
                    (0, 255, 0),
                    2,
                )
                cv2.rectangle(
                    frame,
                    eye_boxes["right_eye"]["top_left"],
                    eye_boxes["right_eye"]["bottom_right"],
                    (0, 255, 0),
                    2,
                )
                for x, y in landmarks:
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), 1)
        return frame
        pass
