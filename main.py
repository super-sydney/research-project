import logging

from evaluation import Evaluation
from feature_extraction import *
from similarity import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting the evaluation")
    m = GenericFourierDescriptor()
    c = Euclidean()

    e = Evaluation(m, c, "dataset/training_subset1")
    score = e.evaluate(visualize=False, save_db=True)
    best_possible_score = e.best_possible_score()

    logging.debug(f"Score: {score}, Best possible score: {best_possible_score}")
    logging.info(f'Ran feature extraction {m} with comparison {c} and got score {score / best_possible_score}')
