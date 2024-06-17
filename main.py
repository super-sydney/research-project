import logging

from evaluation import Evaluation
from feature_extraction import *
from similarity import *

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='main.log', filemode='a')
logger = logging.getLogger(__name__)
logging.getLogger().addHandler(logging.StreamHandler())
if __name__ == "__main__":
    logger.info("Starting the evaluation")
    ms = [Zernike(), BesselFourier(), Legendre(), Tchebichef(), GaborZernike(), GaborLegendre()]
    c = Euclidean()

    for m in ms:
        e = Evaluation(m, c, "dataset/eval_all/both")
        mAP, res = e.evaluate(visualize=False, save_db=False)

        # Add mAP to the results
        res['mAP'] = mAP

        res.to_csv(f'evaluation/results/{m}Both.csv')
        logging.info(f'Ran feature extraction {m} with comparison {c} and got mAP {mAP}')
