import sys
import xml.etree.ElementTree as ET


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
    tree.write("ibug_300W_large_face_landmark_dataset/eyes_only.xml")


if __name__ == "__main__":
    print(sys.argv)
    only_eyes(sys.argv[1])
