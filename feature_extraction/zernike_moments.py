from feature_extraction.ExtractionStrategy import ExtractionStrategy


class ZernikeMoments(ExtractionStrategy):
    """
    Zernike Moments model
    """

    def run(self, image: [[bool]]) -> [float]:
        pass

    def __str__(self):
        return "Zernike Moments"
