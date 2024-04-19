import cv2 as cv
from inspect import getframeinfo, currentframe
import pyautogui as pg
from LK_Optical_flow import LK, Shi_Tomasi
# import matplotlib.pyplot as plt


def heat_map():
    raise NotImplementedError


def get_roi(x: int, y: int, w: int, h: int, image, show: bool = False):
    if show:
        cv.imshow("Original Image", image)
        cv.imshow("sub image", image[y : y + h, x : x + w])
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        return image[y : y + h, x : x + w]


def move_mouse(x, y):
    print("moving mouse to: ", x, y)
    pg.moveTo(x, y)


# Load the cascade classifiers for face and eye detection
capture = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier(
    "G:\Мой диск\Diplom\.venv\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml"
)
# face_cascade = cv.CascadeClassifier(
#     r"C:\Users\scher\AppData\Local\Programs\Python\Python39\Lib\site-packages\cv\data\haarcascade_frontalface_default.xml"
# )
eye_cascade = cv.CascadeClassifier(
    "G:\Мой диск\Diplom\.venv\Lib\site-packages\cv2\data\haarcascade_eye.xml"
)
# eye_cascade = cv.CascadeClassifier(
#     r"C:\Users\scher\AppData\Local\Programs\Python\Python39\Lib\site-packages\cv\data\haarcascade_eye.xml"
# )

while 1:
    ret, img = capture.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = img[y : y + h, x : x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for ex, ey, ew, eh in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv.imshow("img", img)
    k = cv.waitKey(30) & 0xFF
    if k == 27:
        break

capture.release()
cv.destroyAllWindows()
