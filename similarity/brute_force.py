import cv2

from similarity.Similarity import Similarity


class BruteForce(Similarity):
    """
    Brute force comparison
    """

    def compare(self, d1, d2):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(d1, d2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])

        return len(good_matches)
