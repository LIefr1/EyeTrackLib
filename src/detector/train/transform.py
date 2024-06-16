import torch
import imutils
import random
import numpy as np
from math import sin, cos, radians
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, normalize, resize, crop


class Transforms:
    def __init__(self):
        pass

    def _rotate(self, image, landmarks, angle):
        torch.backends.cudnn.allow_tf32 = True
        angle = random.uniform(-angle, +angle)

        transformation_matrix = torch.tensor(
            [
                [+cos(radians(angle)), -sin(radians(angle))],
                [+sin(radians(angle)), +cos(radians(angle))],
            ]
        )

        image = imutils.rotate(np.array(image), angle)

        landmarks = landmarks - 0.5
        new_landmarks = np.matmul(landmarks, transformation_matrix)
        new_landmarks = new_landmarks + 0.5
        return Image.fromarray(image), new_landmarks

    def _resize(self, image, landmarks, img_size):
        image = resize(image, img_size)
        return image, landmarks

    def _color_jitter(self, image, landmarks):
        color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        image = color_jitter(image)
        return image, landmarks

    def _crop_face(self, image, landmarks, crops):
        left = int(crops["left"])
        top = int(crops["top"])
        width = int(crops["width"])
        height = int(crops["height"])

        image = crop(image, top, left, height, width)

        img_shape = np.array(image).shape
        landmarks = torch.tensor(landmarks) - torch.tensor([[left, top]])
        landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])
        return image, landmarks

    def __call__(self, image, landmarks, crops):
        image = Image.fromarray(image)
        image, landmarks = self._crop_face(image, landmarks, crops)
        image, landmarks = self._resize(image, landmarks, (224, 224))
        image, landmarks = self._color_jitter(image, landmarks)
        image, landmarks = self._rotate(image, landmarks, angle=10)

        image = to_tensor(image)
        image = normalize(image, [0.5], [0.5])
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # image = image.to(device)
        # landmarks = landmarks.view(landmarks.size(0), -1).to(device)
        return image, landmarks
