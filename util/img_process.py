""" Preprocess the image before doing fitting job.
"""
from __future__ import absolute_import, division, print_function
import cv2
import numpy as np

LOW_LANE_COLOR = np.uint8([[[0,0,0]]])
UPPER_LANE_COLOR = np.uint8([[[0,0,0]]]) + 40


def standard_preprocess(img, crop=True, down=True, f=True, binary=True):
    """ Perform filter operations to pre-process the image.
    """
    if crop:
        img = crop_image(img, 0.45, 0.85)
    if down:
        img = down_sample(img, (160, 48))
    if f:
        img = lane_filter(img, LOW_LANE_COLOR, UPPER_LANE_COLOR)
    if binary:
        img = img / 255
    return img


def lane_filter(img, lower_lane_color, upper_lane_color):
    """ Use color filter to show lanes in the image.
    """
    laneIMG = cv2.inRange(img, lower_lane_color, upper_lane_color)
    return laneIMG


def crop_image(img, lower_bound, upper_bound):
    img_cropped = img[int(img.shape[0]*lower_bound):int(img.shape[0]*upper_bound),:]
    return img_cropped

def img_load(path):
    img = cv2.imread(path)
    return img

def img_save(img, path):
    cv2.imwrite(path, img)


def down_sample(img, target_size):
    """ Downsample the image to target size.
    """
    if img.shape[0] <= target_size[0] or img.shape[1] <= target_size[1]:
        print('target size should be small than original size')
        return img
    else:
        return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)


def plot_line(image, pts, color='yellow'):
    """ Plot fitting lines on an image.
    """
    if color == 'yellow':
        cv2.polylines(image, [pts], False, (0, 255, 255), 1)
    return image


def enlarge_img(img, times):
    cv2.resize(image, (0,0), fx=times, fy=times)
    return img
