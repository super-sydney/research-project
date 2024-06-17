import numpy as np
from numpy import ndarray
from scipy.special import jn_zeros, jv

from feature_extraction.ExtractionStrategy import ExtractionStrategy


class BesselFourier(ExtractionStrategy):
    def run(self, image: ndarray) -> ndarray:
        """
        Generate the Bessel-Fourier moments of the image.
        :param image: The image to extract the features from
        :return: The features extracted from the image
        """

        return np.abs(self.bessel_fourier_moments(image, 8)).flatten()

    def bessel_fourier_moments(self, image: ndarray, max_order: int) -> float:
        """
        Calculate moments up to the given orders.
        :param image: The image to calculate the moment of
        :param max_order: The maximum order of the moments
        :return: The Legendre moments
        """
        N = np.max(image.shape)
        # convert cartesian coordinates to polar coordinates
        x = np.array(N * [np.array(range(-N // 2, N // 2))])
        y = np.array(N * [np.array(range(N // 2 - 1, -N // 2 - 1, -1))]).T

        r = np.sqrt(x ** 2 + y ** 2)
        r = r / np.max(r)
        theta = np.arctan2(y, x)

        lambda_n = jn_zeros(1, max_order + 1)
        jv_n = jv(2, lambda_n)
        a_n = 1 / (np.pi * (jv_n * jv_n))

        moments = np.zeros((max_order + 1, 2 * max_order + 2), dtype=np.complex128)

        for i in range(1, max_order + 1):
            # Calculate the polynomial for unique r values, then map the polynomial to the r values
            unique_r = np.unique(r)
            temp = jv(1, unique_r * lambda_n[i])
            r_dict = {unique_r[i]: temp[i] for i in range(len(unique_r))}

            polynomial = np.vectorize(lambda x: r_dict[x])(r)
            for j in range(-max_order, max_order + 1):
                if i + abs(j) <= max_order:
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
