import cv2 as cv
import numpy as np
import torch
import logging
from PIL import Image
from ..landmark_detector.model import LandmarkModel
from torchvision.transforms.functional import resize, to_tensor, normalize
import time


class Predictor:
    def __init__(self, model, path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model, path)
        self.model = self.model.to(self.device)

    def _load_model(self, model, path):
        try:
            model.load_state_dict(torch.load(path, map_location=self.device))
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise Exception("Failed to load model")
        return model

    def _preprocess(self, x, y, w, h, gray):
        image = gray[y : y + h, x : x + w]
        image = resize(Image.fromarray(image), size=(224, 224))
        image = to_tensor(image)
        image = normalize(image, [0.5], [0.5])
        return image.unsqueeze(0).to(self.device)

    def predict(self, gray, face):
        x, y, w, h = face
        self.model.eval()
        with torch.no_grad():
            input_tensor = self._preprocess(x, y, w, h, gray)
            landmarks = self.model(input_tensor)
            shape = landmarks.shape[1] // 2
            landmarks = (landmarks.view(shape, 2).cpu().detach().numpy() + 0.5) * np.array(
                [[w, h]]
            ) + np.array([[x, y]])
        return landmarks
