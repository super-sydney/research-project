import cv2
import numpy as np
from numpy import ndarray

from feature_extraction.ExtractionStrategy import ExtractionStrategy
from feature_extraction.moments.legendre import Legendre


class GaborLegendre(ExtractionStrategy):
    """
    Gabor-Legendre features
    """

    def run(self, image: ndarray) -> ndarray:
        """
        Generate the Gabor-Legendre features of the image.
        :param image: The image to extract the features from
        :return: The features extracted from the image
        """

        features = []
        m = Legendre()

        for theta in range(4):
            theta = np.pi * theta / 8.0
            for sigma in np.arange(1, 5):
                for lambd in np.arange(1, 5):
                    kernel = cv2.getGaborKernel((21, 21), sigma, theta, lambd, 0.5, 0, ktype=cv2.CV_64F)
                    filtered = cv2.filter2D(image, -1, kernel)

                    # Calculate the Zernike moments of the filtered image
                    moments = m.run(filtered)
                    features.extend(moments)

        return np.array(features)

    def __str__(self):
        return "GaborLegendre"
