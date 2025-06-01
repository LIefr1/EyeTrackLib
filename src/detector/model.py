from torchvision.models import  resnet152,ResNet152_Weights
import torch.nn as nn
import torch
from typing import Literal

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
    
class EnhancedLandmarkModel(nn.Module):
    def __init__(
        self,
        num_classes=40,
        resnet_model=resnet152(weights=ResNet152_Weights.DEFAULT,)
    ):
        super().__init__()
        self.model = resnet_model
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.se_block = SEBlock(64)  # Adding SE Block
        print ("self.model.fc.in_features", self.model.fc.in_features)
        print ("num_classes", num_classes)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes),

        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print("model", self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("x.shape", x.shape)
        x = x.to(self.device)
        x = self.model(x)
        # x = self.se_block(x)  # Apply SE block
        return self.model.fc(x)

    def get_current_device(self) -> None:
        print("Device is:", self.device)
        
        
        
        
