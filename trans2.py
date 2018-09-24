# python 2.7
from __future__ import absolute_import, division, print_function
import os.path
from os.path import join
from glob import glob

import numpy as np

import util


def convert(img):
    h, w, c = img.shape
    for row in range(h):
        for col in range(w):
            if not (img[row][col] == np.array([255, 255, 255])).all():
                img[row][col] = np.array([0, 0, 255])


if __name__ == '__main__':
    images = util.get_images_from(join('.', 'data', 'training', 'aug-label'))
    num = len(images)
    for i in range(num):
        convert(images[i])
    util.put_images_to(join('.', 'data', 'training', 'aug-label-2'), images, list(range(1, num + 1)))
