import torch.nn as nn
import time
import numpy as np
import torch
import logging
from PIL import Image
from torchvision.transforms.functional import resize, to_tensor, normalize
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from typing import Literal
from src.detector.model import EnhancedLandmarkModel
import cv2 as cv
# Assuming LandmarkModel is already defined and imported


def test_landmark_model():
    # Create a model instance
    num_classes = 40
    model_name = "resnet152"
    model = LandmarkModel(num_classes=num_classes, resnet_model=resnet152())
    # Print the model architecture
    print(model)

    # Generate random 3D data (batch_size, channels, depth, height, width)
    # Example: batch of 4 samples, 1 channel, depth of 16, height of 64, width of 64
    batch_size = 4
    channels = 1
    height = 16
    width = 16
    input_data = torch.randn(batch_size, channels, height, width)

    # Move model to the appropriate device
    model.to(model.device)

    # Perform a forward pass
    output = model(input_data)

    print("Input shape:", input_data.shape)
    print("Input:", input_data[0][0])

    # Print output shape
    print("Output shape:", output.shape)
    print("Output:", output)

    # Check if output shape is as expected (batch_size, num_classes)
    assert output.shape == (
        batch_size,
        num_classes,
    ), f"Expected output shape: {(batch_size, num_classes)}, but got: {output.shape}"

    print("Test passed!")


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
        try:
            x, y, w, h = face
            self.model.eval()
            with torch.no_grad():
                input_tensor = self._preprocess(x, y, w, h, gray)
                start_time = time.time()
                landmarks = self.model(input_tensor)
                end_time = time.time()
                elapsed_time = end_time - start_time
                shape = landmarks.shape[1] // 2
                landmarks = (landmarks.view(shape, 2).cpu().detach().numpy() + 0.5) * np.array(
                    [[w, h]]
                ) + np.array([[x, y]])
            return landmarks, elapsed_time
        except Exception as e:
            print(e)

        return np.empty((68, 2))


import cv2 as cv
import numpy as np
from torchvision.models import resnet152

def run_test_case():
    predictor = Predictor(
        EnhancedLandmarkModel(num_classes=40,
                              resnet_model=resnet152(weights=None)),       
        path="models/EnhancedLandmarkModel-2025_06_16_23:15.pth"
    )

    img_path = "datasets/ibug/ibug/image_031_mirror.jpg"
    frame      = cv.imread(img_path)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    gt = np.array([
                    [308,385],
[703,639],
[268,301],
[455,500],
[479,513],
[325,326],
[341,312],
[374,304],
[414,314],
[384,335],
[354,340],
[537,282],
[564,255],
[600,234],
[642,239],
[621,266],
[577,281],
[482,568],
[365,336],
[578,269],
    ], dtype=np.float32)

    pred, elapsed = predictor.predict(frame_gray, [ 290, 154,481,493])
    pred = np.array(pred, dtype=np.float32)

    mse  = np.mean((pred - gt) ** 2)
    mae  = np.mean(np.abs(pred - gt))
    rmse = np.sqrt(mse)

    print(f"Predicted  :\n{pred}")
    print(f"Elapsed    : {elapsed:.3f} s")
    print(f"MAE   ={mae:.3f}, RMSE  ={rmse:.3f}")

    out = frame.copy()
    for (x, y) in gt.astype(int):
        cv.circle(out, (x, y), radius=3, color=(0,255,0), thickness=-1)
    for (x, y) in pred.astype(int):
        cv.circle(out, (x, y), radius=3, color=(0,0,255), thickness=-1)

    x, y, w, h = [ 290, 154,481,493]
    cv.rectangle(out, (x, y), (x+w, y+h), (255,0,0), 1)

    out_path = "landmark_comparison.jpg"
    cv.imwrite(out_path, out)
    print(f"Saved overlay to {out_path}")





if __name__ == "__main__":
    run_test_case()



