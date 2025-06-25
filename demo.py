from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import QThread, pyqtSignal, Qt
import numpy as np
import cv2 as cv
from collections import deque

from src.tracker_core.tracker import Tracker 
from src.detector.model import EnhancedLandmarkModel
from torchvision.models import resnet18, resnet101, resnet152, resnet50, resnet34


class HeatmapDrawer:
    def __init__(self, width, height, decay=0.9, radius=5):
        self.width = width
        self.height = height
        print(self.width, self.height)
        self.heatmap = np.zeros((height, width), dtype=np.float32)
        self.decay = decay
        self.radius = radius
        self.gaze_points = deque(maxlen=50)

    def update(self, points):
        self.heatmap *= self.decay
        for pt in points:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < self.width and 0 <= y < self.height:
                cv.circle(self.heatmap, (x, y), self.radius, 1, -1)
                self.gaze_points.append((x, y))

    def overlay_on_frame(self, frame):
        norm_map = np.clip(self.heatmap, 0, 1)
        norm_map = (norm_map * 255).astype(np.uint8)
        heat_colored = cv.applyColorMap(norm_map, cv.COLORMAP_JET)
        overlay = cv.addWeighted(frame, 0.7, heat_colored, 0.3, 0)
        return overlay

    def get_current_gaze(self):
        if self.gaze_points:
            return np.mean(self.gaze_points, axis=0)
        return None


class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)

    def __init__(self, cap, tracker):
        super().__init__()
        self.cap = cap
        self.tracker = tracker
        self.ThreadActive = False
        self.heatmap = HeatmapDrawer(
            width=int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        )

    def run(self):
        ret, previous_frame = self.cap.read()
        if not ret:
            print("Error: Unable to read from camera")
            return

        previous_gray = cv.cvtColor(previous_frame, cv.COLOR_BGR2GRAY)

        self.ThreadActive = True
        while self.ThreadActive:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Unable to read frame")
                break

            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            largest_face = self.tracker.get_faces(frame_gray, get_largest_face=True)
            p0 = self.tracker.detect_landmarks(frame_gray, largest_face)

            if p0.size > 0:
                try:
                    new, old = self.tracker.calculate_LK(previous_gray, frame_gray, p0)
                    self.heatmap.update(new.reshape(-1, 2))
                except Exception as e:
                    print(f"Error calculating optical flow: {e}")
                previous_gray = frame_gray.copy()

            frame = self.heatmap.overlay_on_frame(frame)
            qt_img = self.__convert_to_qImage(frame)
            self.ImageUpdate.emit(qt_img)

    def __convert_to_qImage(self, frame):
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        flipped = cv.flip(image, 1)
        return QImage(
            flipped.data,
            flipped.shape[1],
            flipped.shape[0],
            QImage.Format.Format_RGB888
        ).scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)

    def stop(self):
        self.ThreadActive = False
        self.quit()
        self.wait() # Ensure the thread finishes


# --- Main Application Window ---
class Demo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gaze Heatmap Demo")
        self.setGeometry(100, 100, 700, 500)

        self.VBL = QVBoxLayout()

        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)

        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.start_camera)
        self.VBL.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False) # Initially disabled
        self.VBL.addWidget(self.stop_button)

        self.setLayout(self.VBL)

        self.cap = None
        self.tracker =  Tracker(
            model=EnhancedLandmarkModel(resnet_model=resnet152(), num_classes=40),
            path="models/EnhancedLandmarkModel-2025_06_16_23:15.pth",
            lk_params=dict(
                winSize=(60, 60),
                maxLevel=4,
                criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
            ),
        )
        
        #self.mouse = Mouse()
        self.worker = None

    def start_camera(self):
        if self.cap is None:
            self.cap = cv.VideoCapture(0) 
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 2560.0)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080.0)
            if not self.cap.isOpened():
                print("Error: Could not open video stream.")
                self.cap = None
                return

        self.worker = Worker1(self.cap, self.tracker)
        self.worker.ImageUpdate.connect(self.image_update_slot)
        self.worker.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_camera(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.FeedLabel.clear()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def image_update_slot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = Demo()
    window.show()
    sys.exit(app.exec())