import logging

from evaluation import Evaluation
from feature_extraction import *
from similarity import *

logging.basicConfig(level=logging.INFO, )

if __name__ == "__main__":
    logging.info("Starting the evaluation")
    m = ZernikeMoments()
    c = Euclidean()

    e = Evaluation(m, c, "dataset/harmonized_training")
    score = e.evaluate()
    logging.info(f'Ran model {m} with comparison {c} and got score {score}')
