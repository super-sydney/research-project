import numpy as np
from numpy import ndarray

from feature_extraction.ExtractionStrategy import ExtractionStrategy
from .generic_fourier_descriptors.gfd import gfd


class GenericFourierDescriptor(ExtractionStrategy):
    """
    Zernike Moments model
    """

    def run(self, image: ndarray) -> ndarray:
        """
        Run the Zernike Moments model on the image.
        :param image: The image to run the model on
        :return: The features extracted from the image
        """

        X = gfd(image.clip(0, 1), 3, 6)

        return np.abs(X).flatten()

    def __str__(self):
        return "Generic Fourier Descriptor"
