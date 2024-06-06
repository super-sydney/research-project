import numpy as np
from numpy import ndarray

from feature_extraction.ExtractionStrategy import ExtractionStrategy


class Legendre(ExtractionStrategy):
    """
    Gabor model
    """

    def run(self, image: ndarray) -> ndarray:
        """
        Run the Dual Hahn moments on the image.
        :param image: The image to run the model on
        :return: The features extracted from the image
        """

        moments = []
        maxorder = 24

        for p in range(1, maxorder):
            for q in range(1, maxorder):
                if p + q <= maxorder:
                    moments.append(self.legendre_moment(image, p, q))

        return np.array(moments)

    def legendre_moment(self, image: ndarray, p: int, q: int) -> float:
        """
        Calculate the Legendre moment of order p+q.
        :param image: The image to calculate the moment of
        :param p: The p-th moment
        :param q: The q-th moment
        :return: The Legendre moment
        """
        N = np.max(image.shape)
        x = (2 * np.array(range(N)) / N) - 1

        T_p = np.array(N * [self.legendre_polynomial(p, x)]).T
        T_q = np.array(N * [self.legendre_polynomial(q, x)])

        return (2 * p + 1) * (2 * q + 1) / 4 * np.sum(image * T_p * T_q)

    def legendre_polynomial(self, n, x) -> ndarray:
        """
        Calculate the Chebyshev polynomial of the given order.
        :param n: The order of the polynomial
        :param x: The values to calculate the polynomial for
        :return: The Chebyshev polynomial
        """

        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return x
        else:
            prev = self.legendre_polynomial(n - 1, x)
            prev_prev = self.legendre_polynomial(n - 2, x)
            return (1 / n) * ((2 * n - 1) * x * prev - (n - 1) * prev_prev)

    def __str__(self):
        return "Legendre"


if __name__ == "__main__":
    m = Legendre()
    x = np.array(np.arange(-1, 1.01, 0.05))

    import matplotlib.pyplot as plt

    n = 6
    for i in range(0, n):
        y = (m.legendre_polynomial(i, x))
        plt.plot(x, y)

    plt.legend([f"n={i}" for i in range(0, n)])
    plt.show()
