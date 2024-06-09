import sys
import xml.etree.ElementTree as ET
import numpy as np


def only_eyes(file_path: str):
    eyes = set(
        [
            "37",
            "38",
            "39",
            "40",
            "42",
            "41",
            "43",
            "44",
            "45",
            "46",
            "47",
            "48",
            "01",
            "17",
            "32",
            "31",
            "36",
            "09",
            "68",
            "69",
        ]
    )
    tree = ET.parse(file_path)
    root = tree.getroot()
    delete = []

    for box in root.findall("./images/image/box"):
        parts = box.findall(".//part[@name]")
        for part in parts:
            if part.get("name") not in eyes:
                box.remove(part)
    tree.write("datasets/ibug/eyes_only.xml")


def calc(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def add_pupil(file_path: str):
    tree = ET.parse(file_path)
    root = tree.getroot()
    eye_pts = ["37", "38", "40", "41", "43", "44", "47", "48"]
    for box in root.findall("./images/image/box"):
        parts = box.findall(".//part[@name]")
        for part in parts:
            if part.get("name") == "37":
                C = box.findall(".//part[@name='40']")
                x1, y1 = int(part.get("x")), int(part.get("y"))
                x2, y2 = int(C[0].get("x")), int(C[0].get("y"))
                scalar = calc(x1, y1, x2, y2) / 2
                centre = (x1 + scalar, y1 + scalar)

                new_part = ET.Element(
                    "part", dict(name="68", x=str(int(centre[0])), y=str(int(centre[1])))
                )
                box.insert(len(parts), new_part)
                print("centre: ", centre, "x1: ", x1, "y1: ", y1, "x2: ", x2, "y2: ", y2)

            if part.get("name") == "43":
                C = box.findall(".//part[@name='47']")
                x1, y1 = int(part.get("x")), int(part.get("y"))
                x2, y2 = int(C[0].get("x")), int(C[0].get("y"))
                scalar = calc(x1, y1, x2, y2) / 2
                centre = (x1 + scalar, y1 + scalar)

                new_part = ET.Element(
                    "part", dict(name="69", x=str(int(centre[0])), y=str(int(centre[1])))
                )
                box.insert(len(parts) + 1, new_part)
                print("centre: ", centre, "x1: ", x1, "y1: ", y1, "x2: ", x2, "y2: ", y2)

    tree.write("datasets/ibug/pupils.xml")


if __name__ == "__main__":
    print(sys.argv)
    only_eyes(sys.argv[1])
    # add_pupil(sys.argv[1])
