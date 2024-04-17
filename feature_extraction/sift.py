from feature_extraction.ExtractionStrategy import ExtractionStrategy


class SIFT(ExtractionStrategy):
    """
    SIFT model
    """

    def run(self, image: [[bool]]):
        pass

    def __str__(self):
        return "SIFT"
