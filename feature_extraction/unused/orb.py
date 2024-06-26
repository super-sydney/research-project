import cv2
from numpy import ndarray

from feature_extraction.ExtractionStrategy import ExtractionStrategy


class ORB(ExtractionStrategy):
    """
    ORB
    """

    def run(self, image: ndarray) -> ndarray:
        """
        Generate the ORB features of the image.
        :param image: The image to extract the features from
        :return: The features extracted from the image
        """

        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(image, None)

        return des

    def __str__(self):
        return "ORB"
