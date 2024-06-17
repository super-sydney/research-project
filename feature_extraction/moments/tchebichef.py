import numpy as np
from numpy import ndarray

from feature_extraction.ExtractionStrategy import ExtractionStrategy


class Tchebichef(ExtractionStrategy):
    """
    Tchebichef moments
    """

    def run(self, image: ndarray) -> ndarray:
        """
        Generate the Tchebichef moments of the image.
        :param image: The image to extract the features from
        :return: The moments extracted from the image
        """

        moments = []
        max_order = 12

        for p in range(1, max_order):
            for q in range(1, max_order):
                if p + q <= max_order:
                    moments.append(self.tchebichef_moment(image, p, q))

        return np.array(moments)

    def tchebichef_moment(self, image: ndarray, p: int, q: int) -> float:
        """
        Calculate the Tchebichef moment of the image.
        :param image: The image to calculate the moment of
        :param p: The p-th moment
        :param q: The q-th moment
        :return: The Chebyshev moment
        """
        rows, cols = image.shape
        x = np.array(range(rows))
        y = np.array(range(cols))

        T_p = np.array(cols * [self.tchebichef_polynomial(p, x)]).T
        T_q = np.array(rows * [self.tchebichef_polynomial(q, y)])

        return np.sum(image * T_p * T_q)

    def tchebichef_polynomial(self, n: int, x: ndarray) -> ndarray:
        """
        Calculate the Tchebichef polynomial of the given order.
        :param n: The order of the polynomial
        :param x: The values to calculate the polynomial for
        :return: The Tchebichef polynomial
        """
        N = len(x)
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return (2 * x + 1 - N) / N
        else:
            t1 = self.tchebichef_polynomial(1, x)
            t_prev = self.tchebichef_polynomial(n - 1, x)
            t_prev_prev = self.tchebichef_polynomial(n - 2, x)
            t_n = ((2 * n - 1) * t1 * t_prev - (n - 1) * (1 - ((n - 1) * (n - 1)) / (N * N)) * t_prev_prev) / n
            return t_n

    def __str__(self):
        return "Tchebichef"
