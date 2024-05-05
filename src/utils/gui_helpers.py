import pyautogui as pg
import ctypes as ct


def move_mouse(x, y):
    print("moving mouse to: ", x, y)
    pg.moveTo(x, y)


def move_mouse_ct(x: int, y: int):
    print("moving mouse to: ", x, y)
    ct.windll.user32.SetCursorPos(x, y)
    # click_if_stable_params(x=x, y=y)


def click_if_stable_params(x, y, threshold=5, interval=1):
    """
    Makes a click if position have not significantly changed in some time

    :param x: current x coord
    :param y: current y coord
    :param threshold: threshold in pixels, default 5
    :param interval: interval in seconds, default 1
    """
    import time
    import ctypes

    user32 = ctypes.WinDLL("user32")
    GetCursorPos = user32.GetCursorPos
    GetCursorPos.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    GetCursorPos.restype = ctypes.c_int

    while True:
        new_x = ctypes.c_int()
        new_y = ctypes.c_int()
        GetCursorPos(ctypes.byref(new_x), ctypes.byref(new_y))
        if (
            x == 0
            and y == 0
            or (x - new_x.value) ** 2 + (y - new_y.value) ** 2 > threshold**2
        ):
            ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0)
            # Reset reference position after click
            x = new_x.value
            y = new_y.value
        time.sleep(interval)
