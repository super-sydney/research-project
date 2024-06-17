import cv2

from similarity.Similarity import Similarity


class Euclidean(Similarity):
    """
    Euclidean distance comparison
    """

    def compare(self, v1: [float], v2: [float]) -> float:
        dist = cv2.norm(v1, v2, cv2.NORM_L2)
        return 1 / (1 + dist)

    def __str__(self):
        return "Euclidean"
