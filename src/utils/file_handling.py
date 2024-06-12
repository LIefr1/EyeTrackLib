import os
import random
import shutil
import xml.etree.ElementTree as ET


class FileHandling:
    def __init__(self):
        pass

    @staticmethod
    def copy_random_images(source_dir, dest_dir, num_images):
        """
        Copies a random selection of images from the source directory to the destination directory.

        Args:
        source_dir: Path to the directory containing the images to copy from.
        dest_dir: Path to the directory where the random images will be copied to.
        num_images: Number of random images to copy.
        """
        # Check if source directory exists
        if not os.path.isdir(source_dir):
            raise ValueError(f"Source directory {source_dir} does not exist.")

        # Create destination directory if it doesn't exist
        os.makedirs(dest_dir, exist_ok=True)

        # Get all image files from the source directory
        image_files = [
            f
            for f in os.listdir(source_dir)
            if os.path.isfile(os.path.join(source_dir, f))
            and f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        # Check if there are enough images
        if len(image_files) < num_images:
            raise ValueError(f"Source directory has less than {num_images} images.")

        # Randomly select the images to copy
        selected_images = random.sample(image_files, num_images)

        # Copy the selected images
        for image in selected_images:
            source_path = os.path.join(source_dir, image)
            dest_path = os.path.join(dest_dir, image)
            shutil.copy2(
                source_path, dest_path
            )  # Use shutil.copy2 to preserve creation/modification times

        print(f"Successfully copied {num_images} random images to {dest_dir}.")

    @staticmethod
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

        for box in root.findall("./images/image/box"):
            parts = box.findall(".//part[@name]")
            for part in parts:
                if part.get("name") not in eyes:
                    box.remove(part)
        tree.write("datasets/ibug/eyes_only.xml")

    def __calc(x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def add_pupil(self, file_path: str):
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
                    scalar = self.__calc(x1, y1, x2, y2) / 2
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
                    scalar = self.__calc(x1, y1, x2, y2) / 2
                    centre = (x1 + scalar, y1 + scalar)

                    new_part = ET.Element(
                        "part", dict(name="69", x=str(int(centre[0])), y=str(int(centre[1])))
                    )
                    box.insert(len(parts) + 1, new_part)
                    print("centre: ", centre, "x1: ", x1, "y1: ", y1, "x2: ", x2, "y2: ", y2)

        tree.write("datasets/ibug/pupils.xml")
