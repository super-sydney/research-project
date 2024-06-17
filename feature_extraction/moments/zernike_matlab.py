import os

import numpy as np
from numpy import ndarray
from oct2py import octave

from feature_extraction.ExtractionStrategy import ExtractionStrategy


class ZernikeMomentsMatlab(ExtractionStrategy):
    """
    Zernike Moments. Code from Qi et al. MomentToolbox, available at https://github.com/ShurenQi/MomentToolbox
    S. Qi, Y. Zhang, C. Wang, J. Zhou, and X. Cao, “A Survey of Orthogonal Moments for Image Representation: Theory, Implementation, and Evaluation,” ACM Comput. Surv., vol. 55, no. 1, Nov. 2021, doi: 10.1145/3479428.
    """

    def run(self, image: ndarray) -> ndarray:
        """
        Generate the Zernike moments of the image.
        :param image: The image to extract the features from
        :return: The features extracted from the image
        """
        octave.addpath(os.path.join(os.path.dirname(__file__), "matlab"))
        _, X = octave.ZM(image, 8, nout=2)

        return np.abs(X).flatten()

    def __str__(self):
        return "Zernike Moments Matlab"
