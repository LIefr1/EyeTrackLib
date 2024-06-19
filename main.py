from src.detector.model import LandmarkModel
from src.detector.train.trainer import Trainer
from src.detector.train.dataset import Dataset
from src.detector.train.transform import Transforms
from src.tracker_core.tracker import Tracker
from src.mouse_controller.controller import MouseController
import torch.optim as optim
import cv2 as cv
import numpy as np
import sys


def mouse_main():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1200)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 800)
    W = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    mouse = MouseController(frame_w=W, frame_h=H)
    tracker = Tracker(
        model=LandmarkModel(model_name="resnet152", num_classes=40),
        path=r"models/resnet152-155-2024-06-10_08-21-07.pth",
        lk_params=dict(
            winSize=(16, 16),
            maxLevel=4,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
        ),
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

        largest_face = tracker.get_faces(frame_gray, get_largest_face=True)

        p0 = tracker.detect_landmarks(frame_gray, largest_face)
        if p0.size > 0:
            try:
                new, old = tracker.calculate_LK(
                    previous_gray,
                    frame_gray,
                    p0,
                )
                x, y = np.max(new, axis=0)
                print("p:", x, y)
                mouse.move_mouse_new(new)
            except Exception as e:
                print(f"Error calculating optical flow: {e}")

            # img = tracker.draw(frame, (new, old), mask)
            previous_gray = frame_gray.copy()
            p0 = new.reshape(-1, 1, 2)

        cv.imshow("Output", img)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


def train():
    model = LandmarkModel(model_name="resnet152", num_classes=40)
    dataset = Dataset(Transforms())
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    trainer = Trainer(model=model, dataset=dataset, optimizer=optimizer, num_epochs=2)
    trainer.train()


if __name__ == "__main__":
    # mouse_main()
    # train()
    import sys
    from src.demo.demo import Demo
    from PyQt6.QtWidgets import QApplication

    App = QApplication(sys.argv)
    Root = Demo()
    Root.show()
    sys.exit(App.exec())
