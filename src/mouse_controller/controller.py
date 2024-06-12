from pynput.mouse import Controller
from ..utils.sensitivity import Sensitivity


class MouseController(Controller):
    def __init__(self, frame_w, frame_h):
        super().__init__()
        self.sensitivity = Sensitivity()
        pass

    def move_mouse(self, x, y):
        x, y = self.apply([x, y], invert_x=True)
        self.position = (x, y)

    def click(self):
        pass

    def right_click(self):
        pass
