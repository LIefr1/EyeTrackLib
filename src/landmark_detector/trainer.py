import torch
import logging
import sys
import time
import numpy as np
import torch.nn as nn
from tqdm import trange


class Trainer:
    def __init__(
        self,
        model=None,
        dataset=None,
        criterion=nn.MSELoss(),
        optimizer=None,
        num_epochs=10,
    ):
        if model is None:
            raise Exception("Network cannot be None")
        self.model = model
        self.model_name = self.model.get_model_name()
        self.num_epochs = num_epochs
        if dataset is None:
            raise Exception("Dataset cannot be None")
        self.dataset = dataset

        self.criterion = criterion
        if optimizer is None:
            raise Exception("Optimizer cannot be None")
        self.optimizer = optimizer

        self.logger = logging.getLogger(__name__)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler = logging.FileHandler(f"logs/train_log_{self.model_name}.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _print_overwrite(self, step, total_step, loss, operation):
        self.logger.info("\r")
        if operation == "train":
            self.logger.info(
                "Train Steps: %d/%d  Loss: %.4f " % (step, total_step, loss)
            )
        else:
            self.logger.info(
                "Valid Steps: %d/%d  Loss: %.4f " % (step, total_step, loss)
            )

        sys.stdout.flush()

    def _split_data(self, ratio=0.8):
        import torch

        len_valid_set = int(0.1 * len(self.dataset))
        len_train_set = len(self.dataset) - len_valid_set

        self.logger.info(f"The length of Train set is {len_train_set}")
        self.logger.info(f"The length of Valid set is {len_valid_set}")

        (
            train_dataset,
            valid_dataset,
        ) = torch.utils.data.random_split(self.dataset, [len_train_set, len_valid_set])

        # shuffle and batch the datasets
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=16, shuffle=True, num_workers=4
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=8, shuffle=True, num_workers=4
        )
        return train_loader, valid_loader, train_dataset, valid_dataset

    def train(
        self,
    ):
        torch.backends.cudnn.benchmark = True
        try:
            train_loader, valid_loader, train_dataset, valid_dataset = self._split_data(
                self.dataset
            )
        except Exception as e:
            self.logger.error(e)

        self.logger.info("Starting Training")

        torch.autograd.set_detect_anomaly(True)
        model = self.model

        loss_min = np.inf

        start_time = time.time()
        for epoch in trange(
            1,
            self.num_epochs + 1,
            # ascii=True,
            unit="epoch",
            desc="Training model: ",
            dynamic_ncols=True,
            position=0,
        ):
            start_epoch = time.time()
            loss_train = 0
            loss_valid = 0
            running_loss = 0

            model.train()
            for step in trange(
                1,
                len(train_loader) + 1,
                # ascii=True,
                unit="step",
                position=1,
                desc="Training: ",
                dynamic_ncols=True,
                leave=False,
            ):
                images, landmarks = next(iter(train_loader))
                images = images.cuda()
                landmarks = landmarks.view(landmarks.size(0), -1).cuda()

                predictions = model(images)

                # clear all the gradients before calculating them
                self.optimizer.zero_grad()

                # find the loss for the current step
                loss_train_step = self.criterion(predictions, landmarks)

                # calculate the gradients
                loss_train_step.backward()

                # update the parameters
                self.optimizer.step()

                loss_train += loss_train_step.item()
                running_loss = loss_train / step

                self.logger.info(
                    f"Step {step}/{len(train_loader)} - Train Loss: {running_loss:.4f}"
                )

            model.eval()
            with torch.no_grad():
                for step in trange(
                    1,
                    len(valid_loader) + 1,  # ascii=True,
                    unit="step",
                    position=1,
                    desc="Validation: ",
                    dynamic_ncols=True,
                    leave=False,
                ):
                    images, landmarks = next(iter(valid_loader))
                    images = images.cuda()
                    landmarks = landmarks.view(landmarks.size(0), -1).cuda()
                    predictions = model(images)

                    # find the loss for the current step
                    loss_valid_step = self.criterion(predictions, landmarks)

                    loss_valid += loss_valid_step.item()
                    running_loss = loss_valid / step

                    self.logger.info(
                        f"Epoch {epoch}/{self.num_epochs} Step {step}/{len(valid_loader)} - Valid Loss: {running_loss:.4f}"
                    )

            loss_train /= len(train_loader)
            loss_valid /= len(valid_loader)
            end_epoch = time.time()

            eta = (end_epoch - start_time) * (self.num_epochs - epoch) / epoch
            self.logger.info("\n--------------------------------------------------")
            self.logger.info(
                f"""Epoch {epoch}/{self.num_epochs} completed in {end_epoch - start_epoch}s 
                 Train Loss: {loss_train:.4f}, Valid Loss: {loss_valid:.4f}, ETA {eta//60}m {eta%60}s"""
            )
            self.logger.info("--------------------------------------------------")

            if loss_valid < loss_min:
                loss_min = loss_valid
                torch.save(model.state_dict(), f"models/{self.model_name}.pth")
                self.logger.info(
                    f"\nMinimum Validation Loss of {loss_min:.4f} at epoch {epoch}/{self.num_epochs}"
                )
                self.logger.info("Model Saved\n at models/{self.model_name}.pth")

        print("Training Complete")
        print(f"Total Elapsed Time : {time.time() - start_time} s")
        self.logger.info("Training Complete")
        self.logger.info(f"Total Elapsed Time : {time.time() - start_time} s")
