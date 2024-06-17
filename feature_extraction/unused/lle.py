import logging

import numpy as np
from numpy import *
from sklearn.manifold import LocallyLinearEmbedding

from feature_extraction.ExtractionStrategy import ExtractionStrategy

logger = logging.getLogger(__name__)


class LLE(ExtractionStrategy):
    """
    Zernike Moments model
    """

    def run(self, images: [ndarray]) -> [ndarray]:
        """
        Run locally linear embedding on the images.
        :param images: The image to run the model on
        :return: The features extracted from the image
        """
        if images[0].ndim != 2:
            logger.error(f"Expected images to be 2D, got {images[0].ndim}D. Make sure needs_training is set to True.")

        images = np.array([image.flatten() for image in images])

        l = LocallyLinearEmbedding(n_neighbors=5, n_components=20)
        return l.fit_transform(images)

    def __str__(self):
        return "Locally Linear Embedding"
