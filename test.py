import torch
import torch.nn as nn
from PIL import Image
from ITracker.ITrackerModel import ITrackerModel
import os
import scipy.io as sio
import torchvision.transforms as transforms

def loadMetadata(filename, silent = False):
    try:
        # http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
        if not silent:
            print('\tReading metadata from %s...' % filename)
        metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(e)
        print('\tFailed to read the meta file "%s"!' % filename)
        return None
    return metadata

class SubtractMean(object):
    """Normalize an tensor image with mean.
    """

    def __init__(self, meanImg):
        self.meanImg = transforms.ToTensor()(meanImg / 255)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """       
        return tensor.sub(self.meanImg)

model = ITrackerModel()
model = torch.nn.DataParallel(model)
print (model)
# model.cuda()
saved = torch.load('models/checkpoint.pth.tar')["state_dict"]
model.load_state_dict(saved)

imSize = (224, 224)

img = Image.open('test/test.jpg').convert('RGB')

faceMean = loadMetadata(os.path.join("./metadata", 'mean_face_224.mat'))['image_mean']
eyeLeftMean = loadMetadata(os.path.join("./metadata", 'mean_left_224.mat'))['image_mean']
eyeRightMean = loadMetadata(os.path.join("./metadata", 'mean_right_224.mat'))['image_mean']
        
transformFace = transforms.Compose([
    transforms.Resize(imSize),
    transforms.ToTensor(),
    SubtractMean(meanImg=faceMean),
])
transformEyeL = transforms.Compose([
    transforms.Resize(imSize),
    transforms.ToTensor(),
    SubtractMean(meanImg=eyeLeftMean),
])
transformEyeR = transforms.Compose([
    transforms.Resize(imSize),
    transforms.ToTensor(),
    SubtractMean(meanImg=eyeRightMean),
])


output = model(img_tensor)
print(output)