import cv2

from similarity.Similarity import Similarity


class BruteForce(Similarity):
    """
    Brute force comparison
    """

    def compare(self, d1, d2):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(d1, d2, k=2)
        return 1 / len(matches)
