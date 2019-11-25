import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


def rect(X, v1, v2, img):
    """
    Finds bbox.

    Parameters:
    X(tuple): coordinates of origin.
    v1: x basis vector.
    v2: y basis vector.
    img(2d array): image.

    Returns:
    2d array: bbox centered at X with w = 2 * v1, h = 2 * v2
    """
    add_v1 = np.add(X, v1)
    sub_v1 = np.subtract(X, v1)
    add_v2 = np.add(X, v2)
    sub_v2 = np.subtract(X, v2)
    rows = [add_v1[0], sub_v1[0], add_v2[0], sub_v2[0]]
    cols = [add_v1[1], sub_v1[1], add_v2[1], sub_v2[1]]
    max_row = max(rows)
    max_col = max(cols)
    min_row = min(rows)
    min_col = min(cols)
    return img[min_row:(max_row + 1), min_col:(max_col + 1)]


def compute_four_descriptor(X, v1, v2, img):
    """
    Return Fourier Transformed Image.

    Parameters:
    X(tuple): coordinates of origin.
    v1/height: x basis magnitude.
    v2/width: y basis magnitude.
    img(2d array): image.

    Returns:
    2d array: bbox centered at X with w = 2 * v1, h = 2 * v2
    """
    # Crop out a bbox
    patch = rect(X, v1, v2, I)
    # Create Gaussian Window
    # TODO
    # Compute Fourier Transform
    discrete_ft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(discrete_ft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    # Plotting the transformed img
    plt.imshow(img, cmap='gray')
    plt.title('Input Image')
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum')
    plt.show()
    return magnitude_spectrum

#def group_patterns(H, img):