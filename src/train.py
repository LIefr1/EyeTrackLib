from landmark_detector.model import Network
from landmark_detector.Landmark_dataset import FaceLandmarksDataset
from landmark_detector.transforms import Transforms
from landmark_detector.trainer import Trainer
import torch.optim as optim
import torch
import logging
import sys
import time
import numpy as np
import torch.nn as nn
from tqdm import trange


if __name__ == "__main__":
    import torch.optim as optim

    model = Network(
        model_name="resnet18",
    )
    dataset = FaceLandmarksDataset(Transforms())
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    trainer = Trainer(model=model, dataset=dataset, optimizer=optimizer, num_epochs=2)
    trainer.train()
