"""
This file contains the ExtractionStrategy class. This class is used to define the models that can be used to
extract features from images.
"""
from abc import (ABC, abstractmethod)

from numpy import ndarray


class ExtractionStrategy(ABC):
    """
    Abstract class for the extraction strategies
    """

    @abstractmethod
    def run(self, image: ndarray) -> ndarray:
        pass

    def __str__(self):
        pass
