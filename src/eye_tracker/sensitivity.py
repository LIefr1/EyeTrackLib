# File path: sensitivity.py


class Sensitivity:
    def __init__(
        self,
        speed: float = 1.0,
    ):
        """
        Initialize the Sensitivity class with default parameters.
        :param speed: The speed sensitivity parameter.
        :param precision: The precision sensitivity parameter.
        """
        self.speed = speed

    def set_sensitivity(self, speed: float, precision: float):
        """
        Adjust the sensitivity parameters.
        :param speed: The new speed sensitivity parameter.
        :param precision: The new precision sensitivity parameter.
        """
        self.speed = speed
        self.precision = precision

    def apply(self, eye_data: list) -> list:
        """
        Apply the sensitivity adjustments to the eye-tracking data.
        :param eye_data: The raw eye-tracking data.
        :return: Adjusted eye-tracking data.
        """
        adjusted_data = eye_data * self.speed

        return adjusted_data
