import cv2
from typing import Union, Literal
from cv2 import UMat


class VideoCapture:
    def __init__(
        self,
        source: int | str = 0,
        fps=30,
        frameSize: Union[int, int] = (2560, 1080),
    ):
        self.cap = cv2.VideoCapture(source)

        self.cap.set(propId=cv2.CAP_PROP_FRAME_WIDTH, value=frameSize[0])
        self.cap.set(propId=cv2.CAP_PROP_FRAME_HEIGHT, value=frameSize[1])
        self.cap.set(propId=cv2.CAP_PROP_FPS, value=fps)

        self.ret, self.frame = self.cap.read()
        self.cam_properties = {
            "CV_CAP_PROP_POS_MSEC": self.cap.get(cv2.CAP_PROP_POS_MSEC),
            "CV_CAP_PROP_FRAME_WIDTH": self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "CV_CAP_PROP_FRAME_HEIGHT": self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "CAP_PROP_FPS": self.cap.get(cv2.CAP_PROP_FPS),
            "CAP_PROP_POS_MSEC": self.cap.get(cv2.CAP_PROP_POS_MSEC),
            "CAP_PROP_FRAME_COUNT": self.cap.get(cv2.CAP_PROP_FRAME_COUNT),
            "CAP_PROP_BRIGHTNESS": self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            "CAP_PROP_CONTRAST": self.cap.get(cv2.CAP_PROP_CONTRAST),
            "CAP_PROP_SATURATION": self.cap.get(cv2.CAP_PROP_SATURATION),
            "CAP_PROP_HUE": self.cap.get(cv2.CAP_PROP_HUE),
            "CAP_PROP_GAIN": self.cap.get(cv2.CAP_PROP_GAIN),
            "CAP_PROP_EXPOSURE": self.cap.get(cv2.CAP_PROP_EXPOSURE),
            "CAP_PROP_CONVERT_RGB": self.cap.get(cv2.CAP_PROP_CONVERT_RGB),
            "CAP_PROP_WHITE_BALANCE_BLUE_U": self.cap.get(
                cv2.CAP_PROP_WHITE_BALANCE_BLUE_U
            ),
            "CAP_PROP_RECTIFICATION": self.cap.get(cv2.CAP_PROP_RECTIFICATION),
            "CAP_PROP_ISO_SPEED": self.cap.get(cv2.CAP_PROP_ISO_SPEED),
            "CAP_PROP_BUFFERSIZE": self.cap.get(cv2.CAP_PROP_BUFFERSIZE),
            "CAP_PROP_AUTO_EXPOSURE": self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            "CAP_PROP_AUTOFOCUS": self.cap.get(cv2.CAP_PROP_AUTOFOCUS),
            "CAP_PROP_FOCUS": self.cap.get(cv2.CAP_PROP_FOCUS),
            "CAP_PROP_PAN": self.cap.get(cv2.CAP_PROP_PAN),
            "CAP_PROP_TILT": self.cap.get(cv2.CAP_PROP_TILT),
            "CAP_PROP_ZOOM": self.cap.get(cv2.CAP_PROP_ZOOM),
            "CAP_PROP_ROLL": self.cap.get(cv2.CAP_PROP_ROLL),
            "CAP_PROP_IRIS": self.cap.get(cv2.CAP_PROP_IRIS),
        }

    def has_next_frame(self) -> bool:
        return self.ret

    def get_grayscale_frame(self, gray=True) -> UMat:
        if gray:
            return cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        return self.frame

    def print_all_props(self) -> None:
        for prop in self.cam_properties:
            print(prop)

    def show_frame(self, gray=True) -> None:
        if gray:
            cv2.imshow("Frame", cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY))
        cv2.imshow("Frame", self.frame)

    def stopCapture(self):
        self.cap.release()
        cv2.destroyAllWindows()
