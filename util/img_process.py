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
    """ Crop image along the height direction.
    """
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
    img = cv2.resize(img, (0,0), fx=times, fy=times)
    return img


def show_img(img, is_bin=False):
    if is_bin:
        # restore
        pass
    cv2.imshow("image", img)
    cv2.waitKey(10)


def calc_fitting_pts(w, x):
    poly_fit = np.poly1d(w)
    y_fitted = poly_fit(x)
    return y_fitted


def mark_image_with_pt(img, pt, color):
    cv2.circle(img, (pt[0], pt[1]), 3, color)


def mark_image_with_parameters(img, parameters, img_height, num_of_p):
    """ Fit the image.
        @returns
            res_img: image with the fitting line
    """
    w_l, w_r, w_m = parameters[0:3], parameters[3:6], parameters[6:9]
    x = np.linspace(0, img_height, num_of_p)
    pts_list = []
    for cur_w in [w_l, w_r, w_m]:
        pts = np.array([calc_fitting_pts(cur_w, x), x], np.int32).transpose()
        pts_list.append(pts)
    cv2.polylines(img, [pts_list[0]], False, (0,255,255), 1)
    cv2.polylines(img, [pts_list[1]], False, (0,255,255), 1)
    cv2.polylines(img, [pts_list[2]], False, (0,255,0), 1)
    return img
