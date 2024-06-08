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
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)

    tracker = Tracker()

    _, previous_frame = cap.read()
    previous_gray = cv.cvtColor(previous_frame, cv.COLOR_BGR2GRAY)

    mask = np.zeros_like(previous_frame)
    frame_count = 0

    while True:
        frame_count += 1

        ret, frame = cap.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = tracker.get_faces(frame)
        if len(faces) > 0:
            p0 = tracker.detect_landmarks(frame_gray, faces[0])
            print("p0:\n", p0)
            new, old = tracker.calculate_LK(
                previous_gray,
                frame_gray,
                p0,
            )
            img = tracker.draw(frame, (new, old), mask)

            cv.imshow("Output", img)

            previous_gray = frame_gray.copy()
            p0 = new.reshape(-1, 1, 2)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


def train():
    model = LandmarkModel(
        model_name="resnet18",
    )
    dataset = Dataset(Transforms())
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    trainer = Trainer(model=model, dataset=dataset, optimizer=optimizer, num_epochs=2)
    trainer.train()


if __name__ == "__main__":
    main()
