import cv2
import torchvision.transforms.functional as TF
from landmark_detector.model import Network
from landmark_detector.train import Trainer
from landmark_detector.Landmark_dataset import FaceLandmarksDataset
from landmark_detector.transforms import Transforms
import torch.optim as optim

from eye_tracker.tracker import Tracker


def run():
    cap = cv2.VideoCapture(0)
    _, first_frame = cap.read()
    tracker = Tracker(prev_frame=first_frame)
    while True:
        ret, frame = cap.read()
        frame = tracker.draw(frame)
        # tracker.optical_flow(frame=frame)

        cv2.imshow("Output", frame)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    run()
