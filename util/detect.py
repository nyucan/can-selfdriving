# python 2.7

import cv2
import numpy as np

import os
from os.path import join
from glob import glob
from detect_peaks import detect_peaks
from debuger import breakpoint
import util


lane_color = np.uint8([[[0,0,0]]])
lower_lane_color1 = lane_color
upper_lane_color1 = lane_color + 10
img_height = 48
img_width = 160
peaks_distance = 55
window_size = int(peaks_distance / 2)


# crop an image
def crop_image(img, lower_bound, upper_bound):
    img_cropped = img[int(img.shape[0]*lower_bound):int(img.shape[0]*upper_bound),:]
    return img_cropped


# use color filter to show lanes in the image
def lane_filter(img, lower_lane_color, upper_lane_color):
    laneIMG = cv2.inRange(img, lower_lane_color, upper_lane_color)
    return laneIMG


# find the centers for two lines
def find_lane_centers(laneIMG_binary):
    # find peaks as the starting points of the lanes (left and right)
    vector_sum_of_lane_marks = np.sum(laneIMG_binary, axis=0)
    peaks = detect_peaks(vector_sum_of_lane_marks, mpd=peaks_distance)
    # we only use the first two peaks as the starting points of the lanes
    if (len(peaks) >= 2):
        peaks = peaks[:2]
        lane_center_left = peaks[0]
        lane_center_right = peaks[1]
    else:
        # set center manually
        lane_center_left = 60
        lane_center_right = 110
    return lane_center_left, lane_center_right


# to find pixels/indices of one of the left and the right lane
# need to call twice, one for left line, and the other for right lane
def find_pixels_of_lane(laneIMG_binary, lane_center, window_size, width_of_laneIMG_binary):
    indices_nonzero = np.nonzero(laneIMG_binary[:,np.max([0, lane_center-window_size]):np.min([width_of_laneIMG_binary, lane_center+window_size])])
    x = indices_nonzero[0]
    # shifted because we are using a part of laneIMG to find non-zero elements
    y = indices_nonzero[1] + np.max([0,lane_center - window_size])
    return x, y


def fit_image(image):
    number_points_for_poly_fit = 50
    x_fitted = np.linspace(0, img_height, number_points_for_poly_fit)
    poly_order = 2

    laneIMG = lane_filter(image, lower_lane_color1, upper_lane_color1)
    laneIMG_binary = laneIMG / 255

    lane_center_left, lane_center_right = find_lane_centers(laneIMG_binary)

    try:
        x_left, y_left = find_pixels_of_lane(laneIMG_binary, lane_center_left, window_size, img_width)
        w_left = np.polyfit(x_left, y_left, poly_order)
        poly_fit_left = np.poly1d(w_left)
        y_left_fitted = poly_fit_left(x_fitted)

        x_right, y_right = find_pixels_of_lane(laneIMG_binary, lane_center_right, window_size, img_width)
        w_right = np.polyfit(x_right, y_right, poly_order)
        poly_fit_right = np.poly1d(w_right)
        y_right_fitted = poly_fit_right(x_fitted)

        # use uint8 to save space
        pts_left = np.array([y_left_fitted, x_fitted], np.uint8).transpose()
        pts_right = np.array([y_right_fitted, x_fitted], np.uint8).transpose()
        # pts_left = np.array([y_left_fitted, x_fitted], np.uint32).transpose()
        # pts_right = np.array([y_right_fitted, x_fitted], np.uint32).transpose()
        cv2.polylines(image, [pts_left], False, (0, 255, 255), 1)
        cv2.polylines(image, [pts_right], False, (0, 255, 255), 1)
    except TypeError as err:
        print('points not enough')
        return image, None, None
    finally:
        return image, pts_left, pts_right


def plot_lines(image, pts_left, pts_right):
    cv2.polylines(image, [pts_left], False, (0, 255, 255), 1)
    cv2.polylines(image, [pts_right], False, (0, 255, 255), 1)
    return cv2.resize(image, (0,0), fx=6, fy=6)
    # return image


def make_decision():
    pass


def mark_images_from(ori_path, dest_path):
    image_paths = glob(os.path.join(ori_path, '*.png'))
    image_paths.sort(key=util.filename_key)
    for i in range(len(image_paths)):
        image = cv2.imread(image_paths[i])
        image, pts_left, pts_right = fit_image(image)
        marked_image = plot_lines(image, pts_left, pts_right)
        cv2.imwrite(join(dest_path, str(i + 1) + '.png'), marked_image)


if __name__ == '__main__':
    mark_images_from('./images/', './output/')
