""" Preprocess the image before doing fitting job.
"""
from __future__ import absolute_import, division, print_function
from math import floor
import cv2
import numpy as np
from time import time

LOW_LANE_COLOR = np.uint8([[[0,0,0]]])
UPPER_LANE_COLOR = np.uint8([[[0,0,0]]]) + 40


def birdeye(img):
    h, w = img.shape[:2]
    src = np.float32([[w, h-10],    # br
                      [0, h-10],    # bl
                      [0, 20],   # tl
                      [w, 20]])  # tr
    dst = np.float32([[w, h],       # br
                      [0, h],       # bl
                      [0, 0],       # tl
                      [w, 0]])      # tr
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped


def binarize(img):
    """
    Convert an input frame to a binary image which highlight as most as possible the lane-lines.
    :param img: input color frame
    :param verbose: if True, show intermediate results
    :return: binarized frame
    """
    h, w = img.shape[:2]
    binary = np.zeros(shape=(h, w), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, thresh=30, maxval=255, type=cv2.THRESH_BINARY_INV)
    binary = np.logical_or(binary, th)
    sobel_mask = thresh_frame_sobel(img, kernel_size=9)
    binary = np.logical_or(binary, sobel_mask)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.erode(binary.astype(np.uint8), kernel, iterations=2)
    binary = cv2.dilate(binary, kernel, iterations=1)
    return binary


def thresh_frame_sobel(frame, kernel_size):
    """ Apply Sobel edge detection to an input frame, then threshold the result.
        @comment: works not very well = =
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mag = np.uint8(sobel_mag / np.max(sobel_mag) * 255)
    _, sobel_mag = cv2.threshold(sobel_mag, 50, 1, cv2.THRESH_BINARY)
    return sobel_mag.astype(bool)


def standard_preprocess(img, crop=True, down=True, f=True, binary=True):
    """ Perform filter operations to pre-process the image.
    """
    if crop:
        # for offline
        img = crop_image(img, 0.45, 0.85)
        # for online
        # img = crop_image(img, 0.40, 0.80)
    if down:
        img = down_sample(img, (160, 48))
    if f:
        img = lane_filter(img, LOW_LANE_COLOR, UPPER_LANE_COLOR)
    if binary:
        img = img / 255
    return img


def red_filter(rgb_img):
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    res1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    # res2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    # res = res1 + res2
    return res1


def get_rectangle(contours):
    areas = [cv2.contourArea(c) for c in contours]
    max_ind = np.argmax(areas)
    cnt = contours[max_ind]
    x, y, w, h = cv2.boundingRect(cnt)
    return x, y, w, h


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


def img_load_from_stream(stream):
    file_bytes = np.asarray(bytearray(stream.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def detect_distance(img):
    """ Detect obstacle based on red pixels on the original image.
    """
    cropped_img = crop_image(img, 0, 0.5)
    down_img = cv2.resize(cropped_img, dsize=None, fx=0.5, fy=0.5)
    red_img = red_filter(down_img)
    cam_dist = 120 # max distance
    try:
        contours, hierarchy = cv2.findContours(red_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = get_rectangle(contours)
        # cv2.rectangle(red_img, (x,y), (x+w,y+h), (255,0,0), 2) # for debug only
        # show_img(red_img)
        # cam_dist = 5400 // w   # distance in cm
        cam_dist = 2700 // w
    except ValueError:
        cam_dist = 120
    finally:
        if cam_dist > 120:
            return 120
        else:
            return cam_dist


def img_save(img, path):
    cv2.imwrite(path, img)


def down_sample(img, target_size):
    """ Downsample the image to target size.
    """
    if img.shape[0] <= target_size[1] or img.shape[1] <= target_size[0]:
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


def show_img(img, is_bin=False, winname="Test", pos=(40, 30)):
    if is_bin:
        # restore
        img *= 255
        # for i in range(len(img)):
        #     if img[i] == 1:
        #         img[i] == (255, 255, 255)
        #     else:
        #         img[i] == (0, 0, 0)
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, pos[0], pos[1])  # Move it to (40,30)
    cv2.imshow(winname, img)
    cv2.waitKey(5)


def calc_fitting_pts(w, x):
    poly_fit = np.poly1d(w)
    y_fitted = poly_fit(x)
    return y_fitted


def mark_image_with_pt(img, pt, color):
    cv2.circle(img, (pt[0], pt[1]), 3, color)


def mark_image_with_line(img, pt_from, pt_to):
    cv2.line(img, pt_from, pt_to, (255, 0, 255), 1)


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
    cv2.polylines(img, [pts_list[0]], False, (0, 255, 255), 1)
    cv2.polylines(img, [pts_list[1]], False, (0, 255, 255), 1)
    cv2.polylines(img, [pts_list[2]], False, (0, 255, 0), 1)
    return img


def compute_debug_image(image, width, height, num_of_pts, cut_pt, paras):
    delta_y = 15
    delta_x = (-paras[14]) * delta_y
    from_pt = (int(cut_pt[0] - delta_x), cut_pt[1] - delta_y)
    to_pt = (int(cut_pt[0] + delta_x), cut_pt[1] + delta_y)
    cw, ch = int(width / 2), int(height / 2)
    debug_img = mark_image_with_parameters(image, paras, height, num_of_pts)
    mark_image_with_pt(debug_img, (cw, ch), (0, 255, 0))
    mark_image_with_pt(debug_img, cut_pt, (0, 255, 255))
    mark_image_with_line(debug_img, from_pt, to_pt)
    return enlarge_img(debug_img, 4)
