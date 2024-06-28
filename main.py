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
    feature_extraction_methods = [Zernike(), BesselFourier(), Legendre(), Tchebichef(), Gabor(), GaborZernike(),
                                  GaborLegendre()]
    similarity_measure = Euclidean()

    for method in feature_extraction_methods:
        e = Evaluation(method, similarity_measure, "dataset/eval_all/both")
        mAP, res = e.evaluate(save_db=False)

        # Add mAP to the results
        res['mAP'] = mAP

        res.to_csv(f'evaluation/results/{method}maybeBoth.csv')
        logging.info(f'Ran feature extraction {method} with comparison {similarity_measure} and got mAP {mAP}')
