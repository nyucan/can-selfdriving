# python 2.7
from __future__ import absolute_import, division, print_function
from os.path import basename
from os.path import join
from os.path import splitext
from glob import glob

import cv2
import numpy as np


def filename_key(x):
    res = int(splitext(basename(x))[0])
    return res


def get_an_image_from(path):
    img = cv2.imread(path)
    return img


def put_an_image_to(path, image):
    cv2.imwrite(path, image)


# read images
def get_images_from(path):
    files_list = glob(join(path, '*.png'))
    # files_list.sort(cmp_filename)
    files_list.sort(key=filename_key)
    img_list = []
    for i in range(len(files_list)):
        # img = cv2.imread(files_list[i], cv2.IMREAD_COLOR)
        img = cv2.imread(files_list[i])
        img_list.append(img)
    return img_list


# write images
def put_images_to(path, images, image_names=None, start_from=1):
    for i in range(len(images)):
        if type(image_names) != type(None):
            cv2.imwrite(join(path, str(image_names[i])), images[i])
        else:
            cv2.imwrite(join(path, str(i + start_from) + '.png'), images[i])


# translate one-hot vectors
def transfer_to_rgb(bin_img):
    """
    Transfer a classification image with one-hot pixels into a RGB image.
    """
    h, w, c = bin_img.shape
    # rgb_img = np.zeros((h, w, 3))
    rgb_img = np.uint8(np.ones((h, w, 3))) * 255
    for i in range(h):
        for j in range(w):
            if np.all(bin_img[i][j] == [1, 0]):
                rgb_img[i][j] = [255, 255, 255]
            elif np.all(bin_img[i][j] == [0, 1]):
                # rgb_img[i][j] = [0, 0, 255]
                rgb_img[i][j] = [0, 0, 0]
    return rgb_img


def find_peaks(arr):
    """ Detect peaks in an np.array.
        @paras np.array arr
        @return np.array: the index of peaks
    """
