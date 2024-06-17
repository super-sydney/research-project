import cv2
import numpy as np
from numpy import ndarray

from feature_extraction.ExtractionStrategy import ExtractionStrategy


class Gabor(ExtractionStrategy):
    """
    Gabor model
    """

    def run(self, image: ndarray) -> ndarray:
        """
        Run the Gabor model on the image.
        :param image: The image to run the model on
        :return: The features extracted from the image
        """

        # Filter the image using Gabor filters, then calculate the mean and standard deviation of each filtered image

        features = []

        for theta in range(4):
            theta = np.pi * theta / 8.0
            for sigma in np.arange(1, 5):
                for lambd in np.arange(1, 5):
                    kernel = cv2.getGaborKernel((21, 21), sigma, theta, lambd, 0.5, 0, ktype=cv2.CV_32F)
                    filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                    mean = np.mean(filtered)
                    std = np.std(filtered)
                    features.append(mean)
                    features.append(std)

        return np.array(features)

    def __str__(self):
        return "Gabor"

# if __name__ == "__main__":
#     # Generate and show the Gabor filter bank
#     filters = []
#
#     for theta in range(4):
#         theta = np.pi * theta / 4.0
#         for sigma in np.arange(1, 5):
#             for lambd in np.arange(0.05, 0.26, 0.05):
#                 kernel = cv2.getGaborKernel((21, 21), sigma, theta, lambd, 0.5, 0, ktype=cv2.CV_32F)
#                 filters.append(kernel)
#
#     import matplotlib.pyplot as plt
#
#     fig, axs = plt.subplots(16, 4)
#     fig.suptitle("Gabor filter bank")
#
#     for i, ax in enumerate(axs.flat):
#         ax.imshow(filters[i], cmap='gray')
#         ax.axis('off')
#
#     plt.show()
