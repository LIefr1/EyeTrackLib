from pynput.mouse import Controller
from ..utils.sensitivity import Sensitivity


class MouseController(Controller):
    def __init__(self, frame_w, frame_h):
        super().__init__()
        self.sensitivity = Sensitivity(frame_h=frame_h, frame_w=frame_w)
        pass

    def move_mouse(self, x, y):
        x, y = self.sensitivity.apply([x, y], invert_x=True)
        self.position = (x, y)

    def move_mouse_new(self, points: list):
        x, y = self.sensitivity.apply(points, invert_x=True)
        self.position = (x, y)
        pass

    def click(self):
        pass

    def right_click(self):
        pass
