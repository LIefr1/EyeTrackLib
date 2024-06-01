import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import os
import cv2
import numpy as np


class FaceLandmarksDataset(Dataset):
    def __init__(
        self, transform=None, path="datasets\ibug\eyes_only.xml", root_dir="ibug"
    ):
        tree = ET.parse(path)
        root = tree.getroot()

        self.image_filenames = []
        self.landmarks = []
        self.crops = []
        self.transform = transform
        self.root_dir = root_dir

        for filename in root[2]:
            self.image_filenames.append(
                os.path.join(self.root_dir, filename.attrib["file"])
            )

            self.crops.append(filename[0].attrib)

            landmark = []
            parts = filename.findall(".//part")
            print(parts)
            for part in parts:
                x_coordinate = int(part.get("x"))
                y_coordinate = int(part.get("y"))
                landmark.append([x_coordinate, y_coordinate])
            self.landmarks.append(landmark)

        self.landmarks = np.array(self.landmarks).astype("float32")

        assert len(self.image_filenames) == len(self.landmarks)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image = cv2.imread(self.image_filenames[index], 0)
        landmarks = self.landmarks[index]

        if self.transform:
            image, landmarks = self.transform(image, landmarks, self.crops[index])

        landmarks = landmarks - 0.5

        return image, landmarks


# if __name__ == "__main__":
#     dataset = FaceLandmarksDataset()
#     print(len(dataset))
