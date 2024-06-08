import cv2 as cv
import numpy as np
import torch
import logging
from PIL import Image
from ..landmark_detector.model import LandmarkModel
from torchvision.transforms.functional import resize, to_tensor, normalize


class Predictor:
    def __init__(
        self,
        model=LandmarkModel(model_name="resnet152"),
        path=None,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model if path is None else self._load_model(model, path)

    def _load_model(self, model, path):
        try:
            model.load_state_dict(torch.load(path, map_location=self.device))
            logging.info("Model loaded successfully")
        except Exception as e:
            raise Exception("Failed to load model")
            print(e)
        finally:
            return model

    def _preprocess(self, x, y, w, h, gray):
        image = gray[y : y + h, x : x + w]
        image = resize(Image.fromarray(image), size=(224, 224))
        image = to_tensor(image)
        image = normalize(image, [0.5], [0.5])
        return image.unsqueeze(0)

    def predict(
        self,
        gray,
        face,
    ):
        # print(face)
        x, y, w, h = face
        self.model.eval()
        all_landmarks = []
        with torch.no_grad():
            landmarks = self.model(self._preprocess(x, y, w, h, gray))
        landmarks = (landmarks.view(68, 2).cpu().detach().numpy() + 0.5) * np.array(
            [[w, h]]
        ) + np.array([[x, y]])
        all_landmarks.append(landmarks)
        return all_landmarks
