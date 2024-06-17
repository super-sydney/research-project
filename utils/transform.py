"""
Transform the original dataset to have different scaling ,translation and rotation.
"""
import os

import cv2
import numpy as np


def rotate_random(image):
    return cv2.warpAffine(image,
                          cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), np.random.randint(0, 360),
                                                  1.0)
                          , (image.shape[1], image.shape[0]))


def shear(image):
    # pad 2d image on all sides by half of the image size
    w = image.shape[1] // 2
    h = image.shape[0] // 2
    image = cv2.copyMakeBorder(image, h, h, w, w, cv2.BORDER_CONSTANT, value=0)

    # Randomly shear the image by at most 15 degrees
    shear_factor_x = np.tan(np.random.uniform(0, 15) * np.pi / 180)
    shear_factor_y = np.tan(np.random.uniform(0, 15) * np.pi / 180)
    shear_matrix = np.array([
        [1 + shear_factor_x * shear_factor_y, shear_factor_x, 0],
        [shear_factor_y, 1, 0]
    ])

    img = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))

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

    return img


if __name__ == '__main__':
    input_folder = "eval_all"

    images = [f for f in os.listdir(os.path.join(os.getcwd(), input_folder)) if
              f.endswith(".png") or f.endswith(".jpg")]

    if not os.path.exists(os.path.join(input_folder, "rotated")):
        os.makedirs(os.path.join(input_folder, "rotated"))
    if not os.path.exists(os.path.join(input_folder, "sheared")):
        os.makedirs(os.path.join(input_folder, "sheared"))
    if not os.path.exists(os.path.join(input_folder, "both")):
        os.makedirs(os.path.join(input_folder, "both"))

    for image in images:
        image_path = os.path.join(input_folder, image)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Rotate the image by a random amount
        rotated = rotate_random(img)
        cv2.imwrite(os.path.join(input_folder, "rotated", image), rotated)

        scaled = shear(img)
        cv2.imwrite(os.path.join(input_folder, "sheared", image), scaled)

        both = shear(rotated)
        cv2.imwrite(os.path.join(input_folder, "both", image), both)
