import cv2
from numpy import ndarray

from feature_extraction.ExtractionStrategy import ExtractionStrategy


class ORB(ExtractionStrategy):
    """
    SIFT model
    """

    def run(self, image: ndarray) -> ndarray:
        """
        Run the ORB model on the image
        :param image: The image to run the model on
        :return: The features extracted from the image
        """

        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(image, None)

        return des

    def __str__(self):
        return "SIFT"
