import mahotas as mh
from numpy import ndarray

from feature_extraction.ExtractionStrategy import ExtractionStrategy


class ZernikeMoments(ExtractionStrategy):
    """
    Zernike Moments model
    """

    def run(self, image: ndarray) -> ndarray:
        """
        Run the Zernike Moments model on the image.
        :param image: The image to run the model on
        :return: The features extracted from the image
        """
        return mh.features.zernike_moments(image, radius=256, degree=8)

    def __str__(self):
        return "Zernike Moments"
