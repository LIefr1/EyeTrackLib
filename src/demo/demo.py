import sys
import numpy as np
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from src.mouse_controller.controller import MouseController
from src.detector.model import LandmarkModel
from src.tracker_core.tracker import Tracker
import cv2 as cv


class Demo(QWidget):
    def __init__(self):
        super().__init__()
        self.cap = cv.VideoCapture(0)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1200)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 800)
        W = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        H = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        self.mouse = MouseController(frame_w=W, frame_h=H)

        self.tracker = Tracker(
            model=LandmarkModel(model_name="resnet152", num_classes=40),
            path=r"models/resnet152-155-2024-06-10_08-21-07.pth",
            lk_params=dict(
                winSize=(244, 244),
                maxLevel=4,
                criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
            ),
        )

        self.VBL = QVBoxLayout()

        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)

        self.startBTN = QPushButton("Start")
        self.CancelBTN = QPushButton("Stop")
        self.startBTN.clicked.connect(self.startFeed)
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)
        self.VBL.addWidget(self.startBTN)

        self.Worker1 = Worker1(self.cap, self.tracker, self.mouse)

        self.setLayout(self.VBL)
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.Worker1.start()

    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def startFeed(self):
        self.Worker1.start()

    def CancelFeed(self):
        self.Worker1.stop()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:  # Correct key namespaceYou can choose any key here
            self.CancelFeed()


class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)

    def __init__(self, cap, tracker, mouse):
        super().__init__()
        self.cap = cap
        self.tracker = tracker
        self.mouse = mouse
        self.ThreadActive = False

    def run(self):
        ret, previous_frame = self.cap.read()
        if not ret:
            print("Error: Unable to read from camera")
            return

        previous_gray = cv.cvtColor(previous_frame, cv.COLOR_BGR2GRAY)
        mask = np.zeros_like(previous_frame)
        frame_count = 0

        self.ThreadActive = True

        while self.ThreadActive:
            frame_count += 1
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Unable to read frame")
                break
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            largest_face = self.tracker.get_faces(frame_gray, get_largest_face=True)

            p0 = self.tracker.detect_landmarks(frame_gray, largest_face)
            if p0.size > 0:
                try:
                    new, old = self.tracker.calculate_LK(
                        previous_gray,
                        frame_gray,
                        p0,
                    )
                    x, y = np.max(new, axis=0)
                    print("p:", x, y)
                    self.mouse.move_mouse_new(new)
                except Exception as e:
                    print(f"Error calculating optical flow: {e}")

                previous_gray = frame_gray.copy()
                p0 = new.reshape(-1, 1, 2)
            self.ImageUpdate.emit(self.__convert_to_qImage(frame))

    def __convert_to_qImage(self, frame):
        Image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        FlippedImage = cv.flip(Image, 1)
        ConvertToQtFormat = QImage(
            FlippedImage.data,
            FlippedImage.shape[1],
            FlippedImage.shape[0],
            QImage.Format.Format_RGB888,
        )
        return ConvertToQtFormat.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)

    def stop(self):
        self.ThreadActive = False
        self.quit()
