"""
This file contains the Evaluation class. This class is used to evaluate the model
by comparing the features of the images with each other.

"""
import logging
import os
import pickle as pkl
import time

import cv2
import pandas as pd

from feature_extraction import *
from similarity import Similarity

logger = logging.getLogger(__name__)


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

    def load_features(self, save_db=False):
        db_name = os.path.join("database",
                               self.extraction_strategy.__str__() + "." + os.path.split(self.images_path)[-1] + ".pkl")
        # load features from file if it exists
        if os.path.exists(db_name):
            logger.info(f"Loading features from {db_name}")
            images = pkl.load(open(db_name, "rb"))
        else:
            logger.info(f"Loading and extracting features from {self.images_path}")

            images = []
            t = time.time_ns()

            # extract features from all images
            for i, image in enumerate(self.images):
                logger.info(f"Loading and extracting features from {image} ({i + 1}/{len(self.images)})")
                image_path = os.path.join(self.images_path, image)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    logger.warning(f"Could not load image {image_path}")
                    continue
                t1 = time.time_ns()
                images.append((image, self.extraction_strategy.run(img)))
                logger.debug(f"Extracted features from {image} in {(time.time_ns() - t1) / 1e6:.2f} ms")

            logger.info(
                f"Extracting features took {(time.time_ns() - t) / 1e6:.2f} ms (avg: {(time.time_ns() - t) / 1e6 / len(images):.2f} ms/image)")

            images.sort(key=lambda x: x[0])

            # save features to file
            if save_db:
                pkl.dump(images, open(db_name, "wb"))
                logger.info(f"Saved features to {db_name}")

        logger.info(f"Loaded and extracted features of {len(images)} images")

        return images

    def evaluate_ranking(self, similarities, group_index, num_retrieved) -> tuple[float, float]:
        """
        Evaluate the ranking of the images from the same group in the similarity ranking.

        The ranking is evaluated by checking how many images from the same group are in the top
        percentage of the ranking. The precision and recall are calculated based on this.

        :param similarities: the similarities of the images with the input image
        :param group_index: the index of the group of images
        :param percentage: the percentage of the ranking to consider

        :return: the precision and recall of the group in the ranking
        """
        # count how many images are in the same group
        group_count = 0
        for filename, _ in similarities:
            if filename.split("_")[0] == group_index:
                group_count += 1

        # count how many images from the same group are in the top percentage of the ranking
        group_in_top = 0
        for i in range(num_retrieved):
            filename, _ = similarities[i]
            if filename.split("_")[0] == group_index:
                group_in_top += 1

        # calculate precision and recall
        precision = group_in_top / num_retrieved
        recall = group_in_top / group_count

        return precision, recall

    def evaluate(self, save_db=False) -> tuple[float, pd.DataFrame]:
        """
        Evaluate the model by comparing the image_features of the image_features with each other.

        For every image, the image_features are computed using the extraction strategy.
        Then, the image_features are compared with every other image using the similarity method.
        The similarities are sorted and the ranking is evaluated by checking how many image_features
        from the same group are in the top percentage of the ranking. The precision and recall
        are calculated based on this.

        file names are in the format "{group_index}_{image_index}.jpg"
        the first number is the index of the group of watermarks it's
        a part of, the second number is the index within that group.

        :param save_db: whether to save the database of image_features to a file

        :return: the mean Average Precision (mAP) and a DataFrame with the results
        """
        # load all image_features
        all_features = self.load_features(save_db)

        # Collect precision and recall of each group, depending on percentage considered retrieved
        # Also calculate the mean Average Precision (mAP)
        results = []
        mAP = 0

        for i, (filename, image_features) in enumerate(all_features):
            logger.info(f"Comparing image_features of {filename} ({i + 1}/{len(all_features)})")
            # compare image_features with every other image
            similarities = [
                (other_filename, self.similarity.compare(image_features, other_features))
                for other_filename, other_features in all_features
                if filename != other_filename
            ]

            # sort similarities
            similarities.sort(key=lambda x: x[1], reverse=True)

            logger.debug(f"Got similarities for {filename}: {similarities}")

            group_index, image_index = filename.split("_")

            AP = 0
            prev_recall = 0
            for i in range(1, len(similarities) + 1):
                precision, recall = self.evaluate_ranking(similarities, group_index, i)
                AP += precision * (recall - prev_recall)
                results.append(
                    {"group": group_index, "image": image_index, "number_retrieved": i, "precision": precision,
                     "recall": recall,
                     "f1": 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0})
                prev_recall = recall
            logger.info(f"Average Precision for {filename}: {AP}")
            mAP += AP

        mAP /= len(all_features)
        return mAP, pd.DataFrame(results)
