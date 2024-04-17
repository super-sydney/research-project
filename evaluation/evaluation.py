"""
This file contains the Evaluation class. This class is used to evaluate the model
by comparing the features of the images with each other.

"""
import os

import cv2

from feature_extraction import *
from similarity import Similarity


class Evaluation:
    def __init__(self, extraction_strategy: ExtractionStrategy, similarity: Similarity, images_path: str):
        """
        Initialize the Evaluation class with the model, comparison method and images to evaluate.
        :param extraction_strategy: the extraction strategy to evaluate
        :param similarity: the similarity method to use
        :param images_path: the path to a folder of images to evaluate
        """
        self.extraction_strategy = extraction_strategy
        self.images_path = images_path
        self.similarity = similarity

        # get all images from the folder that have a .jpg, .jpeg or .png extension
        self.images = [f for f in os.listdir(images_path) if
                       f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]

    def evaluate(self) -> float:
        """
        Evaluate the model by comparing the features of the images with each other.

        For every image, the features are computed using the model and the comparison
        method is used to make a similarity ranking. The score is calculated based
        on where in this ranking the images from the same class are.

        file names are in the format "{group_index}_{image_index}.jpg"
        the first number is the index of the group of watermarks it's
        a part of, the second number is the index within that group.

        :return: the score of the model with the comparison method
        """
        for image in self.images:
            image = cv2.imread(os.path.join(self.images_path, image), cv2.IMREAD_GRAYSCALE)

            # compute features for image
            features = self.extraction_strategy.run(image)

            # compare features with every other image
            similarities = []
            for other_image in self.images:
                other_features = self.extraction_strategy.run(other_image)
                similarity = self.similarity.compare(features, other_features)
                similarities.append((other_image, similarity))

            # sort similarities
            similarities.sort(key=lambda x: x[1], reverse=True)

            score = 0
            for i in range(len(similarities)):
                # if the image is from the same class
                if similarities[i][0].split("_")[0] == self.images[0].split("_")[0]:
                    score += len(similarities) - i

            return score
