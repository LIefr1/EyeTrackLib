import torch
import torch.nn as nn
from src.detector.model import LandmarkModel
# Assuming LandmarkModel is already defined and imported


def test_landmark_model():
    # Create a model instance
    num_classes = 40
    model_name = "resnet152"
    model = LandmarkModel(num_classes=num_classes, model_name=model_name)

    # Print the model architecture
    print(model)

    # Generate random 3D data (batch_size, channels, depth, height, width)
    # Example: batch of 4 samples, 1 channel, depth of 16, height of 64, width of 64
    batch_size = 4
    channels = 1
    height = 224
    width = 224
    input_data = torch.randn(batch_size, channels, height, width)

    # Move model to the appropriate device
    model.to(model.device)

    # Perform a forward pass
    output = model(input_data)

    print("Input shape:", input_data.shape)
    print("Input:", input_data)

    # Print output shape
    print("Output shape:", output.shape)
    print("Output:", output)

    # Check if output shape is as expected (batch_size, num_classes)
    assert output.shape == (
        batch_size,
        num_classes,
    ), f"Expected output shape: {(batch_size, num_classes)}, but got: {output.shape}"

    print("Test passed!")


if __name__ == "__main__":
    test_landmark_model()
