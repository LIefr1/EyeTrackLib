import os
import random
import shutil


class FileHandling:
    def __init__(self):
        pass

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
