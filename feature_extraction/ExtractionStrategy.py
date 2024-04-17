"""
This file contains the Model class. This class is used to define the models that can be used to
extract features from images.
"""
from abc import (ABC, abstractmethod)


class ExtractionStrategy(ABC):
    """
    Abstract class for the models
    """

    @abstractmethod
    def run(self, image: [[bool]]):
        pass

    def __str__(self):
        pass
