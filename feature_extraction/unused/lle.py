import logging

import numpy as np
from numpy import *
from sklearn.manifold import LocallyLinearEmbedding

from feature_extraction.ExtractionStrategy import ExtractionStrategy

logger = logging.getLogger(__name__)


class LLE(ExtractionStrategy):
    """
    Locally Linear Embedding
    """

    def run(self, images: [ndarray]) -> [ndarray]:
        """
        Run locally linear embedding on the images.
        :param images: The images to embed
        :return: The features extracted from the image
        """

        images = np.array([image.flatten() for image in images])

        l = LocallyLinearEmbedding(n_neighbors=5, n_components=20)
        return l.fit_transform(images)

    def __str__(self):
        return "LLE"
