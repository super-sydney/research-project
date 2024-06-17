import numpy as np
from numpy import ndarray
from skimage.feature import local_binary_pattern

from feature_extraction.ExtractionStrategy import ExtractionStrategy


class LBPHistogram(ExtractionStrategy):
    """
    LBP Histogram
    """

    def run(self, image: ndarray) -> ndarray:
        """
        Generate the LBP histogram of the image.
        :param image: The image to extract the features from
        :return: The features extracted from the image
        """

        # Calculate the LBP histogram of the image
        lbp = local_binary_pattern(image, 8, 1, method='uniform')
        hist, _ = np.histogram(lbp, bins=np.arange(0, 255), density=True)

        return hist

    def __str__(self):
        return "LBPHistogram"
