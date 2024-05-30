import numpy as np
from numpy import ndarray

from feature_extraction.ExtractionStrategy import ExtractionStrategy


class Chebyshev(ExtractionStrategy):
    """
    Gabor model
    """

    def run(self, image: ndarray) -> ndarray:
        """
        Run the Chebyshev moments on the image.
        :param image: The image to run the model on
        :return: The features extracted from the image
        """

        moments = np.array([])

        for p in range(1, 8):
            for q in range(1, 8):
                moments = np.append(moments, self.chebyshev_moment(image, p, q))

        return moments

    def chebyshev_moment(self, image: ndarray, p: int, q: int) -> float:
        """
        Calculate the Chebyshev moment of the image.
        :param image: The image to calculate the moment of
        :param p: The p-th moment
        :param q: The q-th moment
        :return: The Chebyshev moment
        """

        rows, cols = image.shape
        x, y = np.meshgrid(range(rows), range(cols), indexing='ij')
        # x = list(range(rows))
        # y = list(range(cols))

        T_p = self.chebyshev_polynomial(p, x, rows)
        T_q = self.chebyshev_polynomial(q, y, cols)

        return np.sum(image * T_p * T_q)

    def chebyshev_polynomial(self, n: int, x: ndarray, N) -> ndarray:
        """
        Calculate the Chebyshev polynomial of the given order.
        :param n: The order of the polynomial
        :param x: The values to calculate the polynomial for
        :return: The Chebyshev polynomial
        """

        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return (2 * x + 1 - N) / N
        else:
            t1 = self.chebyshev_polynomial(1, x, N)
            t_prev = self.chebyshev_polynomial(n - 1, x, N)
            t_prev_prev = self.chebyshev_polynomial(n - 2, x, N)
            t_n = ((2 * n - 1) * t1 * t_prev - (n - 1) * (1 - ((n - 1) * (n - 1)) / (N * N)) * t_prev_prev) / n
            return t_n

    def __str__(self):
        return "Chebyshev"


if __name__ == "__main__":
    m = Chebyshev()
    x = np.array(np.arange(100))

    import matplotlib.pyplot as plt

    for i in range(4, 6):
        y = (m.chebyshev_polynomial(i, x, len(x)))
        plt.plot(x, y)

    plt.legend([f"n={i}" for i in range(4, 6)])
    plt.show()