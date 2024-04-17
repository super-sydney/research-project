"""
Comparison classes for the comparison of vectors and descriptors.
"""

from abc import (ABC)


class Similarity(ABC):
    """
    Abstract class for the comparison classes
    """

    def compare(self, v1, v2):
        pass
