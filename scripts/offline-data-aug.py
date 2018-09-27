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


def filename_key(x):
    res = int(os.path.splitext(os.path.basename(x))[0])
    return res


def augment(images):
    seq = iaa.Sequential([
        # iaa.Fliplr(1), # horizontally flip 100% of the images
        # iaa.Flipud(1),
        iaa.Affine(rotate=(-25, 25), cval=255)
    ])
    images_aug = seq.augment_images(images)
    return images_aug


def augment_data_label(data, label):
    seq = iaa.Sequential([
        # iaa.Fliplr(1), # horizontally flip 100% of the images
        # iaa.Flipud(1),
        iaa.Affine(translate_px=10, cval=255)
    ])
    data_aug = seq.augment_images(data)
    label_aug = seq.augment_images(label)
    return data_aug, label_aug


def put_images_to(path, images):
    counter = 701
    for i in range(len(images)):
        cv2.imwrite(join(path, str(counter) + '.png'), images[i])
        counter += 1


if __name__ == '__main__':
    src_data_dir = join('.', 'dataset', 'data')
    src_label_dir = join('.', 'dataset', 'label')
    out_data_dir = join('.', 'dataset', 'aug-data')
    out_label_dir = join('.', 'dataset', 'aug-label')

    data = util.get_images_from(src_data_dir)
    label = util.get_images_from(src_label_dir)
    # data_aug = augment(data)
    # label_aug = augment(label)
    data_aug, label_aug = augment_data_label(data, label)

    put_images_to(out_data_dir, data_aug)
    put_images_to(out_label_dir, label_aug)
