# python 2.7

from __future__ import absolute_import, division, print_function
import os.path
from os.path import join, basename
from glob import glob

import cv2
import numpy as np
import tensorflow as tf

from util import util

def get_data(data_folder, label_folder, image_h, image_w, norm=False):
    background_color = np.array([255, 255, 255]) # white
    lane_color = np.array([0, 0, 255])           # red

    image_paths = glob(join(data_folder, '*.png'))
    # label_paths = glob(join(label_folder, '*.png'))

    # make sure the label and image are matched
    # image_paths.sort(key=util.filename_key)
    # label_paths.sort(key=util.filename_key)
    image_paths.sort()
    # label_paths.sort()

    images = []    # data
    gt_images = [] # labels

    for image_file_id in range(0, len(image_paths)):
        image_file = image_paths[image_file_id]
        image = cv2.imread(image_file, 3)
        if (norm):
            image = normalize(image)
        images.append(image)

        # for each image in the training set, find the related label
        img_name = basename(image_file)
        # gt_image_file = label_paths[image_file_id]
        gt_image_file = join(label_folder, img_name)
        gt_image = cv2.imread(gt_image_file, 3)
        gt_bg = np.all(gt_image == background_color, axis=2).reshape(image_h, image_w, 1)
        gt_l = np.all(gt_image == lane_color, axis=2).reshape(image_h, image_w, 1)
        gt_image = np.concatenate((gt_bg, gt_l), axis=2)
        gt_images.append(gt_image)
    return np.array(images), np.array(gt_images)


def get_test_data(data_folder, image_h, image_w, norm=False):
    image_paths = glob(os.path.join(data_folder, '*.png'))
    image_paths.sort(key=util.filename_key)
    images = []
    image_names = []
    for image_file_id in range(0, len(image_paths)):
        image = cv2.imread(image_paths[image_file_id])
        if (norm):
            image = normalize(image)
        images.append(image)
        image_names.append(os.path.basename(image_paths[image_file_id]))
    return np.array(images), np.array(image_names)


def calculate_mean():
    image_h, image_w = (48, 160)
    training_data_dir = join('.', 'data', 'training')
    data_folder = join(training_data_dir, 'aug-data')
    label_folder = join(training_data_dir, 'aug-label')
    data, label = get_data(data_folder, label_folder, image_h, image_w)
    rgb_sum = np.array([0, 0, 0], dtype=np.float64)
    n = len(data) * image_h * image_w + 0.0
    for image in data:
        color_sum = np.sum(np.sum(image, 0), 0)
        rgb_sum += color_sum / n
    return rgb_sum


def calculate_std():
    rgb_mean = calculate_mean()
    image_h, image_w = (48, 160)
    training_data_dir = join('.', 'data', 'training')
    data_folder = join(training_data_dir, 'aug-data')
    label_folder = join(training_data_dir, 'aug-label')
    data, label = get_data(data_folder, label_folder, image_h, image_w)
    rgb_std_list = []
    rgb_std = [0, 0, 0]
    n = len(data) * image_h * image_w + 0.0
    for image in data:
        r = image - rgb_mean
        r = r ** 2
        r = r / n
        r = np.sum(np.sum(r, 0), 0)
        rgb_std += r
    return rgb_std ** (0.5)


def normalize(image):
    rgb_mean = np.array([144.62526042, 143.70833333, 139.995442714])
    rgb_std = np.array([49.06974948, 49.35215121, 49.96974726])
    new_image = image - rgb_mean
    new_image = new_image / rgb_std
    return new_image


# TODO
# get tensorflow dataset (for large dataset that cannot be directly read into the memory)
def get_dataset(data_folder, label_folder, image_h, image_w):
    pass
