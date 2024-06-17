import logging

import cv2
import numpy as np
from numpy import copysign, ndarray

from feature_extraction.ExtractionStrategy import ExtractionStrategy

logger = logging.getLogger(__name__)


class HuMoments(ExtractionStrategy):
    """
    Zernike Moments model
    """

    def run(self, image: ndarray) -> ndarray:
        """
        Run the Zernike Moments model on the image.
        :param image: The image to run the model on
        :return: The features extracted from the image
        """
        moments = cv2.HuMoments(cv2.moments(image)).flatten()

        # Rescale the moments to be more similar to each other
        moments = -1 * copysign(1.0, moments) * np.log10(np.abs(moments))
        logger.debug(f"Extracted features: {moments}")
        return moments

    def __str__(self):
        return "Hu Moments"
