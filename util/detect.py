import cv2
import numpy as np

import os
from os.path import join
from glob import glob

from .detect_peaks import detect_peaks
from . import math_support


IMG_HEIGHT, IMG_WIDTH = 48, 160
IMAGE_CENTER = np.int(IMG_WIDTH / 2)
PEAK_DISTANCE = 50
WINDOW_SIZE = int(PEAK_DISTANCE / 2)
# this number determines how long the tangent line is (in pixel)
LENGTH_OF_TANGENT_LINE = 20

# number of points for polynormial fitting
NUMBER_OF_POINTS = 48
POLY_ORDER = 2


class Detector(object):
    def __init__(self):
        # initialize the lane_peaks_previous
        self.left_peak_previous, self.right_peak_previous = 0, IMG_WIDTH
        # to store the previous curves for left and right lane
        self.is_init = True
        self.w_left_previous, self.w_right_previous = np.zeros((3)), np.zeros((3))

    def find_lane_centers(self, laneIMG_binary):
        """ Find the centers for two lines.
        """
        vector_sum_of_lane_marks = np.sum(laneIMG_binary, axis=0)
        peaks = detect_peaks(vector_sum_of_lane_marks, mpd=PEAK_DISTANCE)
        if (peaks.shape[0] == 1):
            # if only one line
            current_peak = peaks[0]
            if (np.abs(current_peak - self.left_peak_previous) <= np.abs(current_peak - self.right_peak_previous)):
                # print('left line remains, right line is missing')
                lane_center_right = None
                lane_center_left = current_peak
            else:
                # print('right line remains, left line is missing')
                lane_center_left = None
                lane_center_right = current_peak
        elif (peaks.shape[0] == 0):
            # no peak is detected
            lane_center_left, lane_center_right = None, None
            return None, None
        else:
            # we only use the first two peaks as the starting points of the lanes
            peaks = peaks[:2]
            lane_center_left = peaks[0]
            lane_center_right = peaks[1]
        if lane_center_left is not None:
            self.left_peak_previous = lane_center_left
        if lane_center_right is not None:
            self.right_peak_previous = lane_center_right
        return lane_center_left, lane_center_right

    def calc_fitting_weights(self, bin_img):
        """ Calculate the polynormial fitting parameters.
            @returns
                w_left:                np.array
                w_right:               np.array
                w_mid:                 np.array
        """
        lane_center_left, lane_center_right = self.find_lane_centers(bin_img)
        w_left, w_right, w_mid = self.w_left_previous, self.w_right_previous, np.zeros(3)
        use_new_left, use_new_right = False, False

        if (lane_center_left is None and lane_center_right is None):
            print('Detect: End of the trial: No lines')
            return  np.zeros(3), np.zeros(3), np.zeros(3)
        else:
            if lane_center_left is not None:
                x_left, y_left = Detector.find_pixels_of_lane(bin_img, lane_center_left, WINDOW_SIZE, IMG_WIDTH)
                try:
                    w_left = np.polyfit(x_left, y_left, POLY_ORDER)
                    use_new_left = True
                    # self.w_left_previous = w_left
                except:
                    print('Detector: Error!!!')
                    w_left = self.w_left_previous
            if lane_center_right is not None:
                x_right, y_right = Detector.find_pixels_of_lane(bin_img, lane_center_right, WINDOW_SIZE, IMG_WIDTH)
                try:
                    w_right = np.polyfit(x_right, y_right, POLY_ORDER)
                    use_new_right = True
                    # self.w_right_previous = w_right
                except:
                    print('Detector: Error!!!')
                    w_right = self.w_right_previous
            # if lane_center_left is not None and lane_center_right is not None:
            THRESHOLD = 1000
            print('diff', math_support.e_diff(w_left, self.w_left_previous), math_support.e_diff(w_right, self.w_right_previous))
            if self.is_init:
                self.is_init = False
                self.w_left_previous = w_left
                self.w_right_previous = w_right
            else:
                if math_support.e_diff(w_left, self.w_left_previous) > THRESHOLD or not use_new_left:
                    w_left = self.w_left_previous
                else:
                    self.w_left_previous = w_left
                if math_support.e_diff(w_right, self.w_right_previous) > THRESHOLD or not use_new_right:
                    w_right = self.w_right_previous
                else:
                    self.w_right_previous = w_right
            w_mid = (w_left + w_right) / 2
        return w_left, w_right, w_mid

    def get_all_parameters(self, image):
        """ Calculate all parameters used for Q-Learning.
            @returns
                w_left:                np.array
                w_right:               np.array
                w_mid:                 np.array
                distance_to_center:    float64
                    y - IMAGE_CENTER
                    the car on the right: negative
                    the car on the left: positive
                distance_at_mid:       float64
                    y - IMAGE_CENTER
                radian_to_center:         float64
                    arctan(distance_at_mid / (IMG_HEIGHT / 2))
                    the car on the right: negative
                    the car on the left: positive
                curvature_at_mid:      float64
                distance_to_tangent:   float64
                angle_of_tan:          float64
            returns `None` if there is no line at all!
        """
        w_left, w_right, w_mid = self.calc_fitting_weights(image)
        x_fitted = np.linspace(0, IMG_HEIGHT, NUMBER_OF_POINTS)
        poly_fit_mid = np.poly1d(w_mid)
        y_mid_fitted = poly_fit_mid(x_fitted)
        y_bottom = np.int(y_mid_fitted[-1])
        distance_to_center = IMAGE_CENTER - y_bottom
        x_mid = np.int(x_fitted[int(NUMBER_OF_POINTS / 2)])
        y_mid = np.int(y_mid_fitted[int(NUMBER_OF_POINTS / 2)])
        distance_at_mid = IMAGE_CENTER - y_mid
        radian_to_center = - math_support.radian(distance_at_mid, IMAGE_CENTER)
        curvature_at_mid = - math_support.curvature(w_mid, x_mid, 2)
        distance_to_tangent, cut_point = Detector.get_distance_2_tan(w_mid)
        angle_of_tangent = Detector.get_angle_of_tan(w_mid, cut_point)
        return w_left, w_right, w_mid, distance_to_center, distance_at_mid, radian_to_center, curvature_at_mid, distance_to_tangent, angle_of_tangent

    def get_wrapped_all_parameters(self, image):
        """ Wrap the parameters.
            @return wrapped_parameters: np.array
                [0:3]  :  w_l
                [3:6]  :  w_r
                [6:9]  :  w_m
                [9]    :  dc
                [10]   :  dm
                [11]   :  radian
                [12]   :  curvature
                [13]   :  distance to tangent
                [14]   :  angle of tangent
        """
        w_l, w_r, w_m, dc, dm, r, c, d2t, aot = self.get_all_parameters(image)
        wrapped_parameters = np.array([], dtype=np.float64)
        wrapped_parameters = np.concatenate((wrapped_parameters, w_l, w_r, w_m, [dc, dm, r, c, d2t, aot]))
        return wrapped_parameters

    @classmethod
    def get_distance_2_tan(cls, w):
        """ Calculate the distance to tangent point.
            pts: [pos_on_width, pos_on_height] (y, x)
        """
        center_pt = (IMG_WIDTH / 2, IMG_HEIGHT / 2)
        x = np.linspace(0, IMG_HEIGHT, NUMBER_OF_POINTS)
        pts = np.array([cls.calc_fitting_pts(w, x), x], np.int32).transpose()
        min_distance = 1000
        min_pt = (0, 0)
        for pt in pts:
            car_is_right = center_pt[0] - pt[0] > 0
            sign = -1
            if not car_is_right:
                sign = 1
            distance = math_support.distance(pt, center_pt)
            if distance ** 2 < min_distance ** 2:
                min_distance = sign * distance
                min_pt = pt
        return min_distance, list(min_pt)

    @classmethod
    def get_angle_of_tan(cls, w, pt):
        """ Calculate the angle of tangent at the point `pt`
            @params
                w
                pt: (w, h)
        """
        return -(w[1] + 2 * w[0] * pt[1])

    @classmethod
    def find_pixels_of_lane(cls, laneIMG_binary, lane_center, window_size, width_of_laneIMG_binary):
        """ Find pixels/indices of one of the left and the right lane
            need to call twice, one for left line, and the other for right lane
        """
        cropped_bin = laneIMG_binary[:,np.max([0, lane_center - window_size]):np.min([width_of_laneIMG_binary, lane_center + window_size])]
        indices_nonzero = np.nonzero(cropped_bin)
        x = indices_nonzero[0]
        # shifted because we are using a part of laneIMG to find non-zero elements
        y = indices_nonzero[1] + np.max([0, lane_center - window_size])
        return x, y

    @classmethod
    def calc_fitting_pts(cls, w, x):
        poly_fit = np.poly1d(w)
        y_fitted = poly_fit(x)
        return y_fitted

