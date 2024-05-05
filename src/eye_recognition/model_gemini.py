import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Literal


class ImageClassifier:
    def __init__(
        self,
        data_dir: Path,
        model_name: Literal[
            "resnet18", "resnet101", "resnet152", "resnet50", "resnet34"
        ] = "resnet18",
        num_classes: int = 4,
        num_epochs: int = 25,
        batch_size: int = 8,
        lr=0.001,
    ):
        """
        Initializes the image classifier.

        Args:
            data_dir (str): Path to the data directory containing 'train' and 'val' folders.
            model_name (str, optional): Name of the pre-trained model to use. Defaults to "resnet18".
            num_classes (int, optional): Number of classes in the dataset. Defaults to 4.
            num_epochs (int, optional): Number of epochs to train for. Defaults to 25.
            batch_size (int, optional): Batch size for training and validation. Defaults to 8.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        """
        self.data_dir = data_dir
        self.model_name = model_name
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transforms = self._get_data_transforms()
        self.image_datasets, self.data_loaders, self.dataset_sizes, self.class_names = (
            self._load_data()
        )
        self.model = self._prepare_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.fc.parameters(), lr=self.lr, momentum=0.9)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    def _get_data_transforms(self):
        """
        Defines data augmentation and normalization transforms.
        """
        data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        }
        return data_transforms

    def _load_data(self):
        """
        Loads image data from the specified directory.
        """
        image_datasets = {
            x: datasets.ImageFolder(
                os.path.join(self.data_dir, x), self.data_transforms[x]
            )
            for x in ["train", "val"]
        }
        data_loaders = {
            x: torch.utils.data.DataLoader(
                image_datasets[x],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=8,
            )
            for x in ["train", "val"]
        }
        dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
        class_names = image_datasets["train"].classes
        return image_datasets, data_loaders, dataset_sizes, class_names

    def _prepare_model(self):
        """
        Loads and prepares the pre-trained model.
        """
        model_ft = getattr(models, self.model_name)(weights="IMAGENET1K_V1")
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
            nn.Conv2d(
                num_ftrs, 256, kernel_size=3, padding=1
            ),  # Add a convolutional layer
            nn.ReLU(inplace=True),
            nn.Conv2d(
                256, (self.num_classes + 4) * 7 * 7, kernel_size=1
            ),  # Output for classes & bbox coordinates
            nn.View(
                -1, self.num_classes, 7, 7
            ),  # Reshape for class & bbox per 7x7 grid
        )
        model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
        model_ft = model_ft.to(self.device)
        return model_ft

    def train(self):
        """
        Trains the model and saves the best performing model.
        """
        best_model = self._train_model(
            self.model, self.criterion, self.optimizer, self.scheduler, self.num_epochs
        )
        return best_model

    def _train_model(self, model, criterion, optimizer, scheduler, num_epochs=25):
        """
        Training loop for the model.
        """
        since = time.time()

        # Create a temporary directory to save training checkpoints
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, "best_model_params.pt")
            torch.save(model.state_dict(), best_model_params_path)
            best_acc = 0.0

            for epoch in range(num_epochs):
                print(f"Epoch {epoch}/{num_epochs - 1}")
                print("-" * 10)

                # Each epoch has a training and validation phase
                for phase in ["train", "val"]:
                    if phase == "train":
                        model.train()  # Set model to training mode
                    else:
                        model.eval()  # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    for inputs, labels in self.data_loaders[phase]:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == "train"):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == "train":
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                    if phase == "train":
                        scheduler.step()

                    epoch_loss = running_loss / self.dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                    print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                    # deep copy the model
                    if phase == "val" and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), best_model_params_path)

                print()

            time_elapsed = time.time() - since
            print(
                f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
            )
            print(f"Best val Acc: {best_acc:4f}")

            # load best model weights
            model.load_state_dict(torch.load(best_model_params_path))
        return model
