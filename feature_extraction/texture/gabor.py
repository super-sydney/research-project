import cv2
import numpy as np
from numpy import ndarray

from feature_extraction.ExtractionStrategy import ExtractionStrategy


class Gabor(ExtractionStrategy):
    """
    Gabor model
    """

    def run(self, image: ndarray) -> ndarray:
        """
        Generate the Gabor features of the image.
        :param image: The image to extract the features from
        :return: The features extracted from the image
        """
        features = []

        for theta in range(4):
            theta = np.pi * theta / 8.0
            for sigma in np.arange(1, 5):
                for lambd in np.arange(1, 5):
                    kernel = cv2.getGaborKernel((21, 21), sigma, theta, lambd, 0.5, 0, ktype=cv2.CV_32F)
                    filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                    mean = np.mean(filtered)
                    std = np.std(filtered)
                    features.append(mean)
                    features.append(std)

        return np.array(features)

    def __str__(self):
        return "Gabor"
