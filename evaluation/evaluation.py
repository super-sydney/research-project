"""
This file contains the Evaluation class. This class is used to evaluate the model
by comparing the features of the images with each other.

"""
import logging
import os
import pickle as pkl

import cv2
import numpy as np

from feature_extraction import *
from similarity import Similarity

logger = logging.getLogger(__name__)


def scoring_function(x):
    """
    The scoring function is used to calculate how good a particular ranking place is.
    The function is such that the score is 1 for first place (x=0), and decreases to 0 until last place (x=1).
    The sigmoid function is used to make the score decrease slowly at first and faster later on.

    :param x: the position in the ranking, rescaled to be between 0 and 1
    """
    # return 2 - (2 / (1 + 2.71828 ** (-10 * x)))
    return 1 if x < 0.1 else 0


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

        # get all images from the folder that have a .jpg, .jpeg extension
        self.images = [f for f in os.listdir(images_path) if
                       f.endswith(".jpg") or f.endswith(".jpeg")]

    def evaluate(self, needs_training=False, save_db=False) -> float:
        """
        Evaluate the model by comparing the features of the images with each other.

        For every image, the features are computed using the model and the comparison
        method is used to make a similarity ranking. The score is calculated based
        on where in this ranking the images from the same group are.

        file names are in the format "{group_index}_{image_index}.jpg"
        the first number is the index of the group of watermarks it's
        a part of, the second number is the index within that group.

        :param needs_training: whether the model needs to be trained. This will give all images at once to the model, instead of one by one.
        :param save_db: whether to save the database of features to a file

        :return: the score of the model with the comparison method
        """
        # load all images
        db_name = os.path.join("database",
                               self.extraction_strategy.__str__() + "." + os.path.split(self.images_path)[-1] + ".pkl")
        if os.path.exists(db_name):
            logger.info(f"Loading features from {db_name}")
            images = pkl.load(open(db_name, "rb"))
        else:
            logger.info(f"Loading and extracting features from {self.images_path}")

            images = []
            if needs_training:
                # load all images at once
                paths, images = zip(
                    *[(image, cv2.imread(os.path.join(self.images_path, image), cv2.IMREAD_GRAYSCALE))
                      for image in
                      self.images])
                images = np.array(images)
                features = self.extraction_strategy.run(images)
                images = [(paths[i], features[i]) for i in range(len(images))]
            else:
                for i, image in enumerate(self.images):
                    logger.info(f"Loading and extracting features from {image} ({i + 1}/{len(self.images)})")
                    image_path = os.path.join(self.images_path, image)
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        logger.warning(f"Could not load image {image_path}")
                        continue
                    images.append((image, self.extraction_strategy.run(img)))

            if save_db:
                images.sort(key=lambda x: x[0])
                pkl.dump(images, open(db_name, "wb"))
                logger.info(f"Saved features to {db_name}")

        logger.info(f"Loaded and extracted features of {len(images)} images")

        score = 0
        for i, (filename, features) in enumerate(images):
            logger.info(f"Comparing features of {filename} ({i + 1}/{len(images)})")
            # compare features with every other image
            similarities = []
            for other_filename, other_features in images:
                if filename is other_filename:
                    continue
                similarity = self.similarity.compare(features, other_features)
                similarities.append((other_filename, similarity))

            # sort similarities
            similarities.sort(key=lambda x: x[1], reverse=True)

            logger.debug(f"Got similarities for {filename}: {similarities}")

            group_index = filename.split("_")[0]
            for i, (filename, _) in enumerate(similarities):
                # if the image is from the same group, add the score
                if filename.split("_")[0] == group_index:
                    score += scoring_function(i / len(similarities))

        return score

    def best_possible_score(self) -> float:
        """
        Calculate the best possible score for the model with the comparison method.

        The best possible score is the score that would be achieved if the model
        was perfect and ranked all of its groups images at the top. This is calculated
        by checking how many images are in each group, and using the scoring function
        to calculate the score for the best possible ranking.

        :return: the best possible score for the model with the comparison method
        """
        # count how many images are in each group
        group_counts = {}
        for image in self.images:
            group_index = image.split("_")[0]
            if group_index in group_counts:
                group_counts[group_index] += 1
            else:
                group_counts[group_index] = 1

        # calculate the score based on the group counts.
        # Every image is compared with every other image in the same group, making for a total of
        # n rankings of n-1 images. The score is calculated based on the position of the images
        # from the same group in the ranking.
        score = 0
        for count in group_counts.values():
            best_group_score = 0
            for i in range(0, count - 1):
                best_group_score += scoring_function(i / (len(self.images) - 1))
            score += count * best_group_score

        return score
