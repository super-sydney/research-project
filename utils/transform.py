"""
Transform the original dataset to have different scaling ,translation and rotation.
"""

import logging
import os

import cv2
import numpy as np


def rotate_random(image):
    return cv2.warpAffine(image,
                          cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), np.random.randint(0, 360),
                                                  1.0)
                          , (image.shape[1], image.shape[0]))


def scale_and_translate(image, scale, tx, ty):
    return cv2.warpAffine(image, np.float32([[scale, 0, tx], [0, scale, ty]]), (image.shape[1], image.shape[0]))


if __name__ == '__main__':
    logger = logging.getLogger(__name__)

    logger.info("Starting the transformation")

    # parser = argparse.ArgumentParser(
    #     description='Transform the original dataset to have different scaling, translation and rotation.')
    # parser.add_argument('input_folder', type=str, help='Folder containing the original dataset')
    #
    # args = parser.parse_args()
    input_folder = "training_subset1"

    images = [f for f in os.listdir(os.path.join(os.getcwd(), input_folder)) if
              f.endswith(".png") or f.endswith(".jpg")]

    for image in images:
        logger.info(f"Transforming {image}")
        image_path = os.path.join(input_folder, image)
        img = cv2.imread(image_path)

        # Rotate the image by a random amount
        rotated = rotate_random(img)
        cv2.imwrite(os.path.join(input_folder, "rotated", image), rotated)

        # Scale the image down by a random amount and translate it so it stays within the frame
        scale = np.random.uniform(0.25, 0.75)
        # original size is 512x512, new is scale*512xscale*512.
        # The most it can be translated in x and y is 512 - scale*512
        tx = np.random.randint(0, 512 - scale * 512)
        ty = np.random.randint(0, 512 - scale * 512)

        scaled = scale_and_translate(img, scale, tx, ty)
        cv2.imwrite(os.path.join(input_folder, "scaled_translated", image), scaled)

        both = scale_and_translate(rotated, scale, tx, ty)
        cv2.imwrite(os.path.join(input_folder, "rotated_scaled_translated", image), both)
