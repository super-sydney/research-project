import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.special import jn_zeros, jv

from feature_extraction.ExtractionStrategy import ExtractionStrategy


class PolarMoment(ExtractionStrategy):
    def run(self, image: ndarray) -> ndarray:
        """
        Run generic polar moments on the image.
        :param image: The image to run the model on
        :return: The features extracted from the image
        """

        return self.moment(image, 32, 32)

    def moment(self, image: ndarray, n: int, m: int) -> float:
        """
        Calculate moments up to the given orders.
        :param image: The image to calculate the moments of
        :param n: The maximum order of the radial moment
        :param m: The maximum order of the angular moment
        :return: The moments of the image in am trix of shape (n, m)
        """
        N = np.max(image.shape)
        # convert cartesian coordinates to polar coordinates
        x = np.array(N * [np.array(range(-N // 2, N // 2))])
        y = np.array(N * [np.array(range(N // 2 - 1, -N // 2 - 1, -1))]).T

        r = np.sqrt(x ** 2 + y ** 2)
        r = r / np.max(r)
        theta = np.arctan2(y, x)

        moments = np.zeros((n, m), dtype=np.complex128)

        for i in range(1, n):
            for j in range(1, m):
                print("Calculating moment", i, j)
                moments[i, j] = self.norm_factor(i)  # normalization factor
                moments[i, j] *= np.sum(image * self.polynomial(r, i) * np.exp(-1j * j * theta))

        return moments

    def norm_factor(self, n):
        """
        Calculate the normalization factor for the moment.
        :param n: The order of the moment
        :return: The normalization factor
        """
        lambda_n = jn_zeros(1, n)[-1]
        a_n = (jv(2, lambda_n) ** 2) / 2
        return 1 / (2 * np.pi * a_n)

    def polynomial(self, r, n):
        """
        Calculate the polynomial of the given order at the given values.
        :param r: The values to calculate the polynomial for
        :param n: The order of the polynomial
        :return: The polynomial
        """
        lambda_n = jn_zeros(1, n)[-1]
        return jv(1, r * lambda_n)

    def reconstruct(self, moments: ndarray, N: int) -> ndarray:
        """
        Reconstruct the image from the moments.
        :param moments: The moments to reconstruct the image from
        :param N: The size of the image
        :return: The reconstructed image
        """
        x = np.array(N * [np.array(range(-N // 2, N // 2))])
        y = np.array(N * [np.array(range(N // 2 - 1, -N // 2 - 1, -1))]).T

        r = np.sqrt(x ** 2 + y ** 2)
        r = r / np.max(r)
        theta = np.arctan2(y, x)

        image = np.zeros((N, N), dtype=np.complex128)
        for n in range(1, moments.shape[0]):
            for m in range(1, moments.shape[1]):
                print("Reconstructing moment", n, m)
                image += moments[n, m] * self.polynomial(r, n) * np.exp(1j * m * theta)

        return np.abs(image)

    def __str__(self):
        return "Generic Polar Moments"


if __name__ == "__main__":
    # Test the model by extracting features and then reconstructing the image
    m = PolarMoment()
    image = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
    moments = m.run(image)

    reconstructed = m.reconstruct(moments, 16)

    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(reconstructed, cmap="gray")

    # Plot the first few polynomials
    r = np.linspace(0, 1, 100)
    for i in range(1, 5):
        y = m.polynomial(r, i)
        ax[1].plot(r, y)

    ax[1].legend([f"n={i}" for i in range(0, 5)])
    plt.show()
