"""
Harmonizes binarized images to have the same resolution and scale. Note that this does not harmonize the rotation.
"""
import argparse
import os

import cv2
import numpy as np


def get_args():
    """
    Parses the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str,
                        default="dataset/386_binarized", help="Path to the input folder of images to be harmonized")
    parser.add_argument("--output_path", type=str,
                        default="dataset/386_harmonized", help="Path to the output folder of harmonized images")
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = get_args()

    folder_path = args.input_path
    output_path = args.output_path

    images = [f for f in os.listdir(folder_path) if os.path.isfile(
        os.path.join(folder_path, f))]

    for image in images:
        print(f"Processing {image}")
        # Load the image
        img = cv2.imread(os.path.join(folder_path, image), cv2.IMREAD_GRAYSCALE)

        # Crop the image
        y_nonzero, x_nonzero = np.nonzero(img)
        x_min, x_max = np.min(x_nonzero), np.max(x_nonzero)
        y_min, y_max = np.min(y_nonzero), np.max(y_nonzero)

        img = img[y_min:y_max, x_min:x_max]

        # Scale the largest side to be 512 pixels
        scale = 512 / max(img.shape)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        # Binarize the image
        img[np.where(img > 127)] = 255
        img[np.where(img <= 127)] = 0

        # Pad the image to be 512x512
        pad_x: int = 512 - img.shape[1]
        pad_y: int = 512 - img.shape[0]
        value: float = 0
        img = cv2.copyMakeBorder(img,
                                 pad_y // 2,
                                 pad_y - (pad_y // 2),
                                 pad_x // 2,
                                 pad_x - (pad_x // 2),
                                 cv2.BORDER_CONSTANT, value)

        # Save the image to a new folder
        path = os.path.join(output_path, image)
        cv2.imwrite(path, img)
