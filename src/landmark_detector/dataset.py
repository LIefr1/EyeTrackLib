import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(
        self, transform=None, path=r"ibug\eyes_only.xml", root_dir=r"B:\EyeTrackLib\ibug\afw"
    ):
        tree = ET.parse(path)
        root = tree.getroot()

        self.image_filenames = []
        self.landmarks = []
        self.crops = []
        self.transform = transform
        self.root_dir = root_dir

        for filename in root[2]:
            image_path = os.path.join(self.root_dir, filename.attrib["file"]).replace("\\", "/")
            if os.path.exists(image_path):
                self.image_filenames.append(image_path)
                self.crops.append(filename[0].attrib)
                landmark = []
                parts = filename.findall(".//part")
                for part in parts:
                    x_coordinate = int(part.get("x"))
                    y_coordinate = int(part.get("y"))
                    landmark.append([x_coordinate, y_coordinate])
                self.landmarks.append(landmark)
            else:
                print(f"Warning: File {image_path} does not exist and will be skipped.")

        self.landmarks = np.array(self.landmarks).astype("float32")
        assert len(self.image_filenames) == len(
            self.landmarks
        ), "Mismatch between images and landmarks count."

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_path = self.image_filenames[index]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Error: Unable to read the image file {image_path}.")

        landmarks = self.landmarks[index]

        if self.transform:
            image, landmarks = self.transform(image, landmarks, self.crops[index])

        landmarks = landmarks - 0.5

        return image, landmarks
