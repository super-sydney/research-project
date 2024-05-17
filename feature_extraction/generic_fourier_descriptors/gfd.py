import numpy as np


def gfd(bw, m, n):
    if bw.dtype != bool:
        raise ValueError('The input image must be of type "bool"')

    if not isinstance(m, int) or not isinstance(n, int) or m < 0 or n < 0:
        raise ValueError('Input arguments M and N must be an integer greater or equal to zero')

    sz = bw.shape

    maxR = 256

    N = sz[0]
    x = np.linspace(-N / 2, N / 2, N)
    y = x
    X, Y = np.meshgrid(x, y)

    radius = np.sqrt(X ** 2 + Y ** 2) / maxR
    theta = np.arctan2(Y, X)
    theta[theta < 0] += 2 * np.pi

    FR = np.zeros((m + 1, n + 1))
    FI = np.zeros((m + 1, n + 1))
    FD = np.zeros((m + 1) * (n + 1))

    i = 0
    for rad in range(m + 1):
        for ang in range(n + 1):
            tempR = bw * np.cos(2 * np.pi * rad * radius + ang * theta)
            tempI = -1 * bw * np.sin(2 * np.pi * rad * radius + ang * theta)
            FR[rad, ang] = np.sum(tempR)
            FI[rad, ang] = np.sum(tempI)

            if rad == 0 and ang == 0:
                FD[i] = np.sqrt((FR[0, 0] ** 2 + FI[0, 0] ** 2)) / (np.pi * maxR ** 2)
            else:
                FD[i] = np.sqrt((FR[rad, ang] ** 2 + FI[rad, ang] ** 2)) / np.sqrt((FR[0, 0] ** 2 + FI[0, 0] ** 2))
            i += 1

    return FD
