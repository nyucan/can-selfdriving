""" Preprocess the image before doing fitting job.
"""
from __future__ import absolute_import, division, print_function
import cv2


def lane_filter(img, lower_lane_color, upper_lane_color):
    """ Use color filter to show lanes in the image.
    """
    laneIMG = cv2.inRange(img, lower_lane_color, upper_lane_color)
    return laneIMG


def crop_image(img, lower_bound, upper_bound):
    img_cropped = img[int(img.shape[0]*lower_bound):int(img.shape[0]*upper_bound),:]
    return img_cropped


def img_save(img, path):
    cv2.imwrite(path, img)


def down_sample(img, target_size):
    """ Downsample the image to target size.
    """
    if img.shape[0] <= target_size[0] or img.shape[1] <= target_size[1]:
        print('target size should be small than original size')
        return img
    else:
        return cv2.resize(img_cropped, target_size, interpolation=cv2.INTER_LINEAR)
