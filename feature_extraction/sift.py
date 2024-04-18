import cv2
import numpy as np
from numpy import ndarray

from feature_extraction.ExtractionStrategy import ExtractionStrategy


class SIFT(ExtractionStrategy):
    """
    SIFT model
    """

    def run(self, image: ndarray) -> ndarray:
        """
        Run the SIFT model on the image
        :param image: The image to run the model on
        :return: The features extracted from the image
        """

        sift = cv2.SIFT.create()
        kp, des = sift.detectAndCompute(image, np.empty((0, 0), np.uint8))
        return des

    def __str__(self):
        return "SIFT"
