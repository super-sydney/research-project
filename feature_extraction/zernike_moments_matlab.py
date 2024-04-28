import os

import numpy as np
from numpy import ndarray
from oct2py import octave

from feature_extraction.ExtractionStrategy import ExtractionStrategy


class ZernikeMomentsMatlab(ExtractionStrategy):
    """
    Zernike Moments model
    """

    def run(self, image: ndarray) -> ndarray:
        """
        Run the Zernike Moments model on the image.
        :param image: The image to run the model on
        :return: The features extracted from the image
        """
        octave.addpath(os.path.join(os.path.dirname(__file__), "matlab"))
        _, X = octave.ZM(image, 8, nout=2)

        return np.abs(X).flatten()

    def __str__(self):
        return "Zernike Moments Matlab"
