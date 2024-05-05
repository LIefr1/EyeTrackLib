import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from torchvision.models import resnet18, resnet101, resnet152, resnet50, resnet34
from typing import Literal


class CustomResnet(nn.Module):
    def __init__(
        self,
        num_classes=4,
        weights=None,
        model_name: Literal[
            "resnet18", "resnet101", "resnet152", "resnet50", "resnet34"
        ] = "resnet18",
        device: Literal["cpu", "cuda"] = "cuda",
    ):
        super(CustomResnet, self).__init__()
        self.device = device
        self.resnet = (
            self.check_model_name(model_name)(weights="IMAGENET1K_V1")
            if weights is None
            else self.check_model_name(model_name)()
        )
        self.features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Conv2d(
                self.features, 256, kernel_size=3, padding=1
            ),  # Add a convolutional layer
            nn.ReLU(inplace=True),
            nn.Conv2d(
                256, (num_classes + 4) * 7 * 7, kernel_size=1
            ),  # Output for classes & bbox coordinates
            nn.View(-1, num_classes, 7, 7),  # Reshape for class & bbox per 7x7 grid
        )

    def forward(self, x):
        self.resnet(x)
        return x

    def parameters(self):
        return self.resnet.fc.parameters()

    @classmethod
    def check_model_name(cls, model_name):
        nets = {
            "resnet18": resnet18(),
            "resnet101": resnet101(),
            "resnet152": resnet152(),
            "resnet50": resnet50(),
            "resnet34": resnet34(),
        }
        return nets.get(model_name)

    def turn_off_grad(self):
        for param in self.resnet.parameters():
            param.requires_grad = False

    @staticmethod
    def base_transforms():
        return {
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

    @staticmethod
    def load_data(data_dir, transforms, batch_size=8, num_workers=8, shuffle=True):
        image_datasets = {
            x: datasets.ImageFolder(os.path.join(data_dir, x), transforms[x])
            for x in ["train", "val"]
        }
        data_loaders = {
            x: torch.utils.data.DataLoader(
                image_datasets[x],
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
            )
            for x in ["train", "val"]
        }
        dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
        class_names = image_datasets["train"].classes

        return {
            "data_loaders": data_loaders,
            "dataset_sizes": dataset_sizes,
            "class_names": class_names,
        }

    @staticmethod
    def imshow(inp, title=None):
        """Display image for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)

    @staticmethod
    def load_model_dict(
        model,
        path,
    ):
        try:
            existing_model = torch.load(path)
            model.load_state_dict(existing_model)
        except FileNotFoundError or Exception as e:
            print(e)
        finally:
            return model

    def train(
        self,
        criterion,
        train_data_loaders: dict,
        optimizer,
        scheduler,
        num_epochs=25,
        path=None,
    ):
        if path is not None:
            model = self.load_model_dict(path, self.resnet)
        else:
            model = self.resnet
        (
            data_loaders,
            dataset_sizes,
        ) = (
            train_data_loaders["data_loaders"],
            train_data_loaders["dataset_sizes"],
        )
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
                    for inputs, labels in data_loaders[phase]:
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

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]

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

    def save_model(self, path="models/eye_model.pt"):
        torch.save(self.resnet.state_dict(), path)

    def visualize_model(
        self,
        num_images=6,
        data_loaders=None,
        class_names=None,
    ):
        was_training = self.resnet.training
        self.resnet.eval()
        images_so_far = 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(data_loaders["val"]):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.resnet(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images // 2, 2, images_so_far)
                    ax.axis("off")
                    ax.set_title(f"predicted: {class_names[preds[j]]}")
                    self.imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        self.resnet.train(mode=was_training)
                        return
            self.resnet.train(mode=was_training)


# from src.eye_recognition.model import CustomResnet
# import torch.nn as nn
# from torch.optim import lr_scheduler
# import torch.optim as optim

# transforms = CustomResnet.base_transforms()

# model = CustomResnet(
#     num_classes=4,
#     weights="IMAGENET1K_V1",
#     model_name="resnet34",
#     device="cuda",
# ).turn_off_grad()


# data = CustomResnet.load_data(
#     data_dir="Eye dataset",
#     transforms=transforms,
#     batch_size=8,
#     num_workers=8,
#     shuffle=True,
# )

# criterion = nn.CrossEntropyLoss()
# optimizer_conv = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


# model.train(
#     criterion, data, optimizer_conv, exp_lr_scheduler, num_epochs=25
# ).save_model()
