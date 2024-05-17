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
        Run the Gabor model on the image.
        :param image: The image to run the model on
        :return: The features extracted from the image
        """

        # Filter the image using Gabor filters, then calculate the mean and standard deviation of each filtered image

        features = []

        for theta in range(4):
            theta = theta / 4. * np.pi
            for sigma in (1, 3):
                for frequency in (0.05, 0.25):
                    kernel = cv2.getGaborKernel((21, 21), sigma, theta, frequency, 0.5, 0, ktype=cv2.CV_32F)
                    filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                    features = np.append(features, [filtered.mean(), filtered.std()])

        return features

    def __str__(self):
        return "Gabor"
