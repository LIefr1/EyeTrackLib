from torchvision.models import resnet18, resnet101, resnet152, resnet50, resnet34
import torch.nn as nn
import torch
from typing import Literal


class LandmarkModel(nn.Module):
    def __init__(
        self,
        num_classes=136,
        model_name: Literal[
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
        ] = "resnet18",
    ):
        super().__init__()
        self.model_name = model_name
        self.model = self._set_model()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _set_model(self):
        models = {
            "resnet18": resnet18(),
            "resnet34": resnet34(),
            "resnet50": resnet50(),
            "resnet101": resnet101(),
            "resnet152": resnet152(),
        }
        return models.get(self.model_name)

    def get_model_name(self) -> str:
        return self.model_name

    def forward(self, x):
        x = x.to(self.device)
        self.model.to(self.device)
        x = self.model(x)
        return x

    def get_current_device(self) -> None:
        print("Curent device is:", self.device)
