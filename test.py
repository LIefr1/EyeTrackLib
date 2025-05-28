import torch
import torch.nn as nn
from PIL import Image
from ITracker.ITrackerModel import ITrackerModel

import torchvision.transforms as transforms



model = ITrackerModel()
model.cuda()
saved = torch.load('models/best_checkpoint.pth.tar')['state_dict']
model.load_state_dict(saved)

img = Image.open('test.jpg').convert('RGB')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # assumes [-1, 1]
])

img_tensor = transform(img).unsqueeze(0).cuda()

output = model(img_tensor)
print(output)