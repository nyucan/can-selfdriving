# python 2.7
from __future__ import absolute_import, division, print_function
import os.path
from os.path import join
from glob import glob
import matplotlib
matplotlib.use('TkAgg')
from imgaug import augmenters as iaa
import cv2
import numpy as np

import util


def augment(images):
    seq = iaa.Sequential([
        # iaa.Fliplr(1), # horizontally flip 100% of the images
        # iaa.Flipud(1), # vertically flip 100% of the images
        iaa.Affine(rotate=(-25, 25), cval=255)    # position shift
        # iaa.WithChannels(0, iaa.Add((10, 100))),  # color shift
        # iaa.WithChannels(1, iaa.Add((10, 100))),  # color shift
        # iaa.WithChannels(2, iaa.Add((10, 100))),  # color shift
        # iaa.GaussianBlur(sigma=(0.0, 3.0)),      # blur
        # iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255))
    ])
    images_aug = seq.augment_images(images)
    return images_aug


if __name__ == '__main__':
    input_dir = join('.', 'data', 'aug', 'input')
    output_dir = join('.', 'data', 'aug', 'output')

    data = util.get_images_from(input_dir)
    data_aug = augment(data)

    util.put_images_to(output_dir, data_aug, start_from=481)
