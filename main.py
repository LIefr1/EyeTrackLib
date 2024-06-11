from src.landmark_detector.model import LandmarkModel
from src.landmark_detector.trainer import Trainer
from src.landmark_detector.dataset import Dataset
from src.landmark_detector.transforms import Transforms
from src.eye_tracker.tracker import Tracker


import torch.optim as optim
import cv2 as cv
import numpy as np
import sys


def main():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = Tracker(
        model=LandmarkModel(model_name="resnet152", num_classes=40),
        path=r"models/resnet152-105-2024-06-10_08-21-07.pth",
    )
    ret, previous_frame = cap.read()
    if not ret:
        print("Error: Unable to read from camera")
        return

    previous_gray = cv.cvtColor(previous_frame, cv.COLOR_BGR2GRAY)
    mask = np.zeros_like(previous_frame)
    frame_count = 0
    img = previous_frame  # Initialize img to avoid referencing before assignment

    while True:
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame")
            break
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = tracker.get_faces(frame_gray)
        if len(faces) > 0:
            p0 = tracker.detect_landmarks(frame_gray, faces[0])
            if p0.size > 0:
                try:
                    new, old = tracker.calculate_LK(
                        previous_gray,
                        frame_gray,
                        p0,
                    )
                    img = tracker.draw(frame, (new, old), mask)
                    previous_gray = frame_gray.copy()
                    p0 = new.reshape(-1, 1, 2)
                except Exception as e:
                    print(f"Error calculating optical flow: {e}")

        cv.imshow("Output", img)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


def train():
    model = LandmarkModel(model_name="resnet18", num_classes=40)
    dataset = Dataset(Transforms())
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    trainer = Trainer(model=model, dataset=dataset, optimizer=optimizer, num_epochs=2)
    trainer.train()


if __name__ == "__main__":
    main()
    # train()
