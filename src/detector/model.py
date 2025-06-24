from torchvision.models import resnet152, ResNet152_Weights
import torch.nn as nn
import torch
from typing import Literal


class EnhancedLandmarkModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 40,
        resnet_model=resnet152(weights=ResNet152_Weights.DEFAULT),
    ):
        super().__init__()
        self.model = resnet_model

        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_input_features = self.model.fc.in_features
        if num_input_features != num_classes:
            self.model.fc = nn.Linear(num_input_features, num_classes)
        else:
            print("Warning: The number of classes matches the ResNet default. The final layer was not replaced.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model loaded on device: {self.device}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("Input shape:", x.shape)
        x = x.to(self.device)

        output = self.model(x)

        return output

    def get_current_device(self) -> None:
        print("Model is on device:", self.device)


