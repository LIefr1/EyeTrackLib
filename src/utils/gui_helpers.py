import pyautogui as pg


def move_mouse(x, y):
    print("moving mouse to: ", x, y)
    pg.moveTo(x, y)
