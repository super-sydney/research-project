import mahotas as mh
from numpy import ndarray

from feature_extraction.ExtractionStrategy import ExtractionStrategy


class Zernike(ExtractionStrategy):
    """
    Zernike Moments
    """

    def run(self, image: ndarray) -> ndarray:
        """
        Generate the Zernike moments of the image.
        :param image: The image to extract the features from
        :return: The features extracted from the image
        """
        return mh.features.zernike_moments(image, radius=256, degree=10)

    def __str__(self):
        return "Zernike"
