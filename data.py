# python 2.7

from __future__ import absolute_import, division, print_function
import os.path
from os.path import join, expanduser
from glob import glob

import scipy.misc
import numpy as np


def gen_batch_function(data_folder, image_h, image_w):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_h, image_w: Tuple - Shape of image
    :return:
    """
    background_color = np.array([255, 255, 255]) # white
    left_lane_color = np.array([255, 0, 0])      # red
    right_lane_color = np.array([0, 0, 255])     # blue

    def get_batches_fn(batch_size):
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = glob(os.path.join(data_folder, 'gt_image_2', '*.png'))

        # make sure the label and image are matched
        image_paths.sort()
        label_paths.sort()

        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file_id in range(batch_i, batch_i + batch_size):
                image_file = image_paths[image_file_id]
                gt_image_file = label_paths[image_file_id]

                image = scipy.misc.imread(image_file, mode='RGB')
                gt_image = scipy.misc.imread(gt_image_file, mode='RGB')

                gt_bg = np.all(gt_image == background_color, axis=2).reshape(image_h, image_w, 1)
                gt_ll = np.all(gt_image == left_lane_color, axis=2).reshape(image_h, image_w, 1)
                gt_rl = np.all(gt_image == right_lane_color, axis=2).reshape(image_h, image_w, 1)
                gt_image = np.concatenate((gt_bg, gt_ll, gt_rl), axis=2)
                images.append(image)
                gt_images.append(gt_image)
            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def get_data(data_folder, image_h, image_w):
    background_color = np.array([255, 255, 255]) # white
    left_lane_color = np.array([255, 0, 0])      # red
    right_lane_color = np.array([0, 0, 255])     # blue

    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
    label_paths = glob(os.path.join(data_folder, 'gt_image', '*.png'))

    # make sure the label and image are matched
    image_paths.sort()
    label_paths.sort()

    images = []    # data
    gt_images = [] # labels

    for image_file_id in range(0, len(image_paths)):
        image_file = image_paths[image_file_id]
        gt_image_file = label_paths[image_file_id]

        image = scipy.misc.imread(image_file, mode='RGB')
        gt_image = scipy.misc.imread(gt_image_file, mode='RGB')

        gt_bg = np.all(gt_image == background_color, axis=2).reshape(image_h, image_w, 1)
        gt_ll = np.all(gt_image == left_lane_color, axis=2).reshape(image_h, image_w, 1)
        gt_rl = np.all(gt_image == right_lane_color, axis=2).reshape(image_h, image_w, 1)
        gt_image = np.concatenate((gt_bg, gt_ll, gt_rl), axis=2)

        images.append(image)
        gt_images.append(gt_image)

    return np.array(images), np.array(gt_images)


def get_test_data(data_folder, image_h, image_w):
    image_paths = glob(os.path.join(data_folder, '*.png'))
    images = []
    image_names = []
    for image_file_id in range(0, len(image_paths)):
        image = scipy.misc.imread(image_paths[image_file_id], mode='RGB')
        images.append(image)
        image_names.append(os.path.basename(image_paths[image_file_id]))
    return np.array(images), np.array(image_names)


# TODO
def preprocess():
    pass


def unit_test():
    pass


if __name__ == '__main__':
    unit_test()
