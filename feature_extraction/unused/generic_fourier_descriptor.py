import numpy as np
from numpy import ndarray

from feature_extraction.ExtractionStrategy import ExtractionStrategy
from .generic_fourier_descriptors.gfd import gfd


class GenericFourierDescriptor(ExtractionStrategy):
    """
    Generic Fourier Descriptors. Code by Frederik Kratzert.
    """

    def run(self, image: ndarray) -> ndarray:
        """
        Extracts the features from the image using the Generic Fourier Descriptors method.
        :param image: The image to extract the features from
        :return: The features extracted from the image
        """

        X = gfd(image.clip(0, 1), 3, 6)

        return np.abs(X).flatten()

    def __str__(self):
        return "Generic Fourier Descriptor"
