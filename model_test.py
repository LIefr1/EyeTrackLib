import os
import sys
import torch
import torch.nn as nn
from src.detector.model import LandmarkModel

# from src.tracker_core.landmark_predictor import Predictor
import time
import numpy as np
import torch
import logging
from PIL import Image
from torchvision.transforms.functional import resize, to_tensor, normalize
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import torch.nn as nn
from typing import Literal

# Assuming LandmarkModel is already defined and imported


def test_landmark_model():
    # Create a model instance
    num_classes = 40
    model_name = "resnet152"
    model = LandmarkModel(
        num_classes=num_classes,
        model_name=model_name,
    )

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


def test_predictor():
    mypath = r"models"
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    for f in onlyfiles:
        print(f)

    for path in onlyfiles[: len(onlyfiles) // 2]:
        model_name = path.split("-")[0]
        print(
            f"---------------------------------------${path}--------------------------------------------"
        )

        fullPath = os.path.join(mypath, path)

        # Test 1: Standard Test Case
        gray_image = np.random.randint(0, 256, (500, 500), dtype=np.uint8)
        face_bbox = (100, 100, 200, 200)
        run_test_case(fullPath, gray_image, face_bbox, model_name)

        # Test 2: Different Image Sizes
        print("Test 2: Different Image Sizes")
        gray_image = np.random.randint(0, 256, (1000, 1000), dtype=np.uint8)
        face_bbox = (200, 200, 400, 400)
        run_test_case(fullPath, gray_image, face_bbox, model_name)

        # Test 3: Small Face Bounding Box
        print("Test 3: Small Face Bounding Box")
        gray_image = np.random.randint(0, 256, (500, 500), dtype=np.uint8)
        face_bbox = (150, 150, 50, 50)
        run_test_case(fullPath, gray_image, face_bbox, model_name)

        # Test 4: Large Face Bounding Box
        print("Test 4: Large Face Bounding Box")
        gray_image = np.random.randint(0, 256, (500, 500), dtype=np.uint8)
        face_bbox = (50, 50, 400, 400)
        run_test_case(fullPath, gray_image, face_bbox, model_name)

        # Test 5: Empty Image
        print("Test 5: Empty Image")
        gray_image = np.zeros((500, 500), dtype=np.uint8)
        face_bbox = (100, 100, 200, 200)
        run_test_case(fullPath, gray_image, face_bbox, model_name)

        # Test 6: No Face Bounding Box
        print("Test 6: No Face Bounding Box")
        gray_image = np.random.randint(0, 256, (500, 500), dtype=np.uint8)
        face_bbox = (0, 0, 0, 0)
        run_test_case(fullPath, gray_image, face_bbox, model_name)


def run_test_case(fullPath, gray_image, face_bbox, model_name):
    # Initialize the predictor
    predictor = Predictor(
        LandmarkModel(model_name=model_name, num_classes=40),
        path=fullPath,
    )

    # Perform prediction
    try:
        landmarks, elapsed_time = predictor.predict(gray_image, face_bbox)

        # Print the predicted landmarks and elapsed time
        print("Predicted Landmarks:", landmarks)
        print("Elapsed Time:", elapsed_time)

        # Dummy ground truth for accuracy calculation
        ground_truth_landmarks = np.random.rand(20, 2) * 200 + np.array([100, 100])

        # Calculate accuracy metrics
        mse = np.mean((landmarks - ground_truth_landmarks) ** 2)
        mae = np.mean(np.abs(landmarks - ground_truth_landmarks))
        rmse = np.sqrt(mse)

        print("Mean Squared Error (MSE):", mse)
        print("Mean Absolute Error (MAE):", mae)
        print("Root Mean Squared Error (RMSE):", rmse)
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    test_predictor()
