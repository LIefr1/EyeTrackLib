import screeninfo
import numpy as np
import sys


class Sensitivity:
    def __init__(self, frame_w, frame_h, speed: float = 1.0, precision: float = 1.0):
        """
        Initialize the Sensitivity class with default parameters.
        :param speed: The speed sensitivity parameter.
        :param precision: The precision sensitivity parameter.
        """
        self.WIDTH = screeninfo.get_monitors()[0].width
        self.HEIGHT = screeninfo.get_monitors()[0].height
        self.FRAME_W = frame_w
        self.FRAME_H = frame_h

        self.speed = speed
        self.precision = precision

    def __invert_x(self, data: list) -> list:
        """
        Invert the x-axis data.
        :param data: The raw eye-tracking data.
        :return: Inverted x-axis eye-tracking data.
        """
        return [[self.WIDTH - x, self.HEIGHT - y] for x, y in data]

    def __translate_coordinates(
        self, x, y, source_width, source_height, target_width, target_height
    ):
        """
        Translate coordinates from one resolution to another.
        :param x: x coordinate.
        :param y: y coordinate.
        :param source_width: Width of the source resolution.
        :param source_height: Height of the source resolution.
        :param target_width: Width of the target resolution.
        :param target_height: Height of the target resolution.
        :return: Translated x and y coordinates.
        """
        S_x = target_width / source_width
        S_y = target_height / source_height

        x_prime = x * S_x
        y_prime = y * S_y

        return x_prime, y_prime

    def __set_speed_by_resolution(self):
        """
        Adjust the speed parameter based on the resolution.
        """
        self.speed = (self.WIDTH / self.FRAME_W + self.HEIGHT / self.FRAME_H) / 2

    def set_sensitivity(self, speed: float, precision: float):
        """
        Adjust the sensitivity parameters.
        :param speed: The new speed sensitivity parameter.
        :param precision: The new precision sensitivity parameter.
        """
        self.speed = speed
        self.precision = precision

    def apply(self, eye_data: list, invert_x=False) -> list:
        """
        Apply the sensitivity adjustments to the eye-tracking data.
        :param eye_data: The raw eye-tracking data.
        :param invert_x: Boolean flag to invert the x-axis data.
        :return: Adjusted eye-tracking data.
        """
        # if invert_x:
        #     eye_data = self.__invert_x(eye_data)

        eye_data = np.array(eye_data)
        pmax = eye_data[8]
        print("pmax", pmax)

        # Adjust speed and precision based on region

        adjusted_data = self.__translate_coordinates(
            x=pmax[0],
            y=pmax[1],
            source_height=self.FRAME_H,
            source_width=self.FRAME_W,
            target_height=self.HEIGHT,
            target_width=self.WIDTH,
        )

        # Adjust the data based on speed
        adjusted_data = [coord * self.speed for coord in adjusted_data]

        return adjusted_data
