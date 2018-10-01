# python 2.7

from __future__ import absolute_import, division, print_function
import os.path
from os.path import join, expanduser
from glob import glob

import cv2
import numpy as np
import tensorflow as tf

from util import util

def get_data(data_folder, label_folder, image_h, image_w):
    background_color = np.array([255, 255, 255]) # white
    lane_color = np.array([0, 0, 255])           # red

    image_paths = glob(os.path.join(data_folder, '*.png'))
    label_paths = glob(os.path.join(label_folder, '*.png'))

    # make sure the label and image are matched
    image_paths.sort(key=util.filename_key)
    label_paths.sort(key=util.filename_key)
    # image_paths.sort()
    # label_paths.sort()

    images = []    # data
    gt_images = [] # labels

    for image_file_id in range(0, len(image_paths)):
        image_file = image_paths[image_file_id]
        gt_image_file = label_paths[image_file_id]

        image = cv2.imread(image_file, 3)
        gt_image = cv2.imread(gt_image_file, 3)

        gt_bg = np.all(gt_image == background_color, axis=2).reshape(image_h, image_w, 1)
        gt_l = np.all(gt_image == lane_color, axis=2).reshape(image_h, image_w, 1)
        gt_image = np.concatenate((gt_bg, gt_l), axis=2)

        images.append(image)
        gt_images.append(gt_image)
    return np.array(images), np.array(gt_images)


def get_test_data(data_folder, image_h, image_w):
    image_paths = glob(os.path.join(data_folder, '*.png'))
    image_paths.sort(key=util.filename_key)
    images = []
    image_names = []
    for image_file_id in range(0, len(image_paths)):
        image = cv2.imread(image_paths[image_file_id])
        images.append(image)
        image_names.append(os.path.basename(image_paths[image_file_id]))
    return np.array(images), np.array(image_names)


# TODO
# get tensorflow dataset (for large dataset that cannot be directly read into the memory)
def get_dataset(data_folder, label_folder, image_h, image_w):
    pass
