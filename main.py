import logging

from evaluation import Evaluation
from feature_extraction import *
from similarity import *

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting the evaluation")
    ms = [Zernike(), Legendre(), Chebyshev(), BesselFourier()]
    c = Euclidean()

    for m in ms:
        e = Evaluation(m, c, "dataset/training_subset/base")
        score, res = e.evaluate(visualize=False, save_db=False)
        best_possible_score = e.best_possible_score()

        res.to_csv(f'evaluation/results/{m}24.csv')
        # logging.debug(f"Score: {score}, Best possible score: {best_possible_score}")
        # logging.info(f'Ran feature extraction {m} with comparison {c} and got score {score / best_possible_score}')
