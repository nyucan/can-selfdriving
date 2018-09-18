# python 2.7
from __future__ import absolute_import, division, print_function

import numpy as np


def transfer_to_rgb(bin_img):
    """
    Transfer a classification image with one-hot pixels into a RGB image.
    """
    h, w, c = bin_img.shape
    rgb_img = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            if np.all(bin_img[i][j] == [1., 0., 0.]):
                rgb_img[i][j] = [255, 255, 255]
            elif np.all(bin_img[i][j] == [0., 1., 0.]):
                rgb_img[i][j] = [255, 0, 0]
            elif np.all(bin_img[i][j] == [0., 0., 1.]):
                rgb_img[i][j] = [0, 0, 255]
    return rgb_img


# TODO
def preprocess():
    pass
