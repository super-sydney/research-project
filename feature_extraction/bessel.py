import time

import cv2
import numpy as np
from numpy import ndarray
from scipy.special import jn_zeros, jv

from feature_extraction.ExtractionStrategy import ExtractionStrategy


class BesselFourier(ExtractionStrategy):
    def run(self, image: ndarray) -> ndarray:
        """
        Run the Bessel-Fourier moments on the image.
        :param image: The image to run the model on
        :return: The features extracted from the image
        """

        return np.abs(self.bessel_fourier_moment(image, 10)).flatten()

    def bessel_fourier_moment(self, image: ndarray, maxorder: int) -> float:
        """
        Calculate moments up to the given orders.
        :param image: The image to calculate the moment of
        :param n: The maximum order of the radial moment
        :param m: The maximum order of the angular moment
        :return: The Legendre moment
        """
        N = np.max(image.shape)
        # convert cartesian coordinates to polar coordinates
        x = np.array(N * [np.array(range(-N // 2, N // 2))])
        y = np.array(N * [np.array(range(N // 2 - 1, -N // 2 - 1, -1))]).T

        r = np.sqrt(x ** 2 + y ** 2)
        r = r / np.max(r)
        theta = np.arctan2(y, x)

        lambda_n = jn_zeros(1, maxorder + 1)
        jv_n = jv(2, lambda_n)
        a_n = 1 / (np.pi * (jv_n * jv_n))

        moments = np.zeros((maxorder + 1, 2 * maxorder + 2), dtype=np.complex128)

        for i in range(1, maxorder + 1):
            # Calculate the polynomial for unique r values, then map the polynomial to the r values
            unique_r = np.unique(r)
            temp = jv(1, unique_r * lambda_n[i])
            r_dict = {unique_r[i]: temp[i] for i in range(len(unique_r))}

            polynomial = np.vectorize(lambda x: r_dict[x])(r)
            for j in range(-maxorder, maxorder + 1):
                if i + abs(j) <= maxorder:
                    moments[i, j] = a_n[i] * np.sum(image * polynomial * np.exp(-1j * j * theta))
        return moments

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
        lmbda = jn_zeros(1, np.max(moments.shape))

        for n in range(1, moments.shape[0]):
            for m in range(1, moments.shape[1]):
                image += moments[n, m] * jv(1, r * lmbda[n]) * np.exp(1j * m * theta)

        return np.abs(image)

    def __str__(self):
        return "BesselFourier"


if __name__ == "__main__":
    t = time.time_ns()
    m = BesselFourier()
    image = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
    moments = m.run(image)

    print((time.time_ns() - t) / 1e6, "ms")
    # reconstructed = m.reconstruct(moments, 256)
    #
    # plt.imshow(reconstructed, cmap="gray")
    #
    # plt.show()

    # v = 1
    # r = np.linspace(0, 1, 100)
    # lambda_n = jn_zeros(v, 8)
    # for i in range(0, 8):
    #     a_n = (jv(v + 1, lambda_n[i]) ** 2) / 2
    #     y = a_n * jv(v, r * lambda_n[i])
    #     plt.plot(r, y)
    #
    # plt.legend([f"n={i}" for i in range(1, 8)])
    # plt.show()
