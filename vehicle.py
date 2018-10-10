# python 2.7
# run on the raspberry pi cars
from __future__ import absolute_import, division, print_function
from os.path import join
import io
import time
import sys
import picamera
import numpy as np
import cv2
import RPi.GPIO as GPIO
from PIL import Image

from util.detect_peaks import detect_peaks
import util.math_support as ms


MARGIN_LEFT = 260
MARGIN_RIGHT = 380
turningKL = 4.2 # for lane tracking, turning left
turningKR = 1   # for lane tracking, turning right

lane_color = np.uint8([[[0,0,0]]])
lower_lane_color1 = lane_color
upper_lane_color1 = lane_color + 40

# the distances of peaks should be tuned
# the peaks indicate the possible center of left and right lanes
peaks_distance = 50

# size after downsampling
size_after_downsampling = (160, 50)

# this is the width of laneIMG
width_of_laneIMG_binary = size_after_downsampling[0]
height_of_laneIMG_binary = size_after_downsampling[1]

# use a window to find all pixels of the left lane and the right lane
# here, the window size is half as the distance between peaks
window_size = int(peaks_distance/2)

# x_axis for polynomial fitting
number_points_for_poly_fit = 50
x_fitted = np.linspace(0, height_of_laneIMG_binary, number_points_for_poly_fit)
# polynomial fitting for lanes
poly_order = 2

# this number determines how long the tangent line is (in pixel)
length_tangent_line = 20

# initialize the lane_peaks_previous
left_peak_previous, right_peak_previous = 0, width_of_laneIMG_binary

# to store the previous curves for left and right lane
w_left_previous, w_right_previous = np.zeros((3)), np.zeros((3))

go_straight_00 = np.array([50, 60])
go_straight_01 = np.array([40, 50])
go_straight_02 = np.array([60, 70])
turn_left_00 = np.array([0, 70])
turn_left_01 = np.array([0, 90])
turn_left_02 = np.array([30, 90])
turn_right_00 = np.array([90, 0])
turn_right_01 = np.array([100, 0])
turn_right_02 = np.array([100, 30])
action_set = np.concatenate([[go_straight_00],
                            [go_straight_01],
                            [go_straight_02],
                            [turn_left_00],
                            [turn_left_01],
                            [turn_left_02],
                            [turn_right_00],
                            [turn_right_01],
                            [turn_right_02]])

dim_action = action_set.shape[0]
# random actions exclude the ones used in simple logic function
rand_action_options = np.array([1, 2, 4, 5, 7, 8])

distance_threshold = 3 # threshold in pixels

# epsilon greedy: epsilon% to choose random action
epsilon_greedy = 0.

# suppose we have 5 states to store: distance_to_center, distance_at_mid, first_order_derivative_at_x, intercept, curvature
##dim_state = 5
dim_state = 3

# threshold_distance_error determines if the distances_to_center is corrupted/wrongly measured
threshold_distance_error = 50

# memory for storing states and actions
memory_size = 1000
memory_counter = 0
memory = np.zeros((memory_size,dim_state+1))

#  -------

distance = []
data = []
stream = io.BytesIO()
distance_to_center_last = 0
distance_at_mid_last = 0
firstStart = True
counter_in_firstStart = 0
kpi_time = 0


def compute_points_on_tangent_line(slope, intercept, x_curv, length_tangent_line):
    """ Compute two points on the tangent line that tangent to the mid_fitted line at x_curv.
    """
    x_curv_1 = x_curv - int(length_tangent_line/2)
    y_curv_1 = int(slope_function(first_order_derivative_at_x, intercept, x_curv_1))
    x_curv_2 = x_curv + int(length_tangent_line/2)
    y_curv_2 = int(slope_function(first_order_derivative_at_x, intercept, x_curv_2))
    return x_curv_1, y_curv_1, x_curv_2, y_curv_2


def motor_init():
    GPIO.setmode(GPIO.BCM)
    ENA = 26
    ENB = 11
    IN1 = 19
    IN2 = 13
    IN3 = 6
    IN4 = 5

    time.sleep(1)
    #  Motor Pins
    GPIO.setup(ENA, GPIO.OUT) #ENA
    GPIO.setup(ENB, GPIO.OUT) #ENB
    GPIO.setup(IN1, GPIO.OUT) #IN1
    GPIO.setup(IN2, GPIO.OUT) #IN2
    GPIO.setup(IN3, GPIO.OUT)  #IN3
    GPIO.setup(IN4, GPIO.OUT)  #IN4

    # PWM pin and Frequency
    pwmR = GPIO.PWM(26, 100)
    pwmL = GPIO.PWM(11, 100)
    pwmR.start(0)
    pwmL.start(0)

    time.sleep(1)
    # Set motor forwards
    GPIO.output(19, GPIO.HIGH)
    GPIO.output(13, GPIO.LOW)
    GPIO.output(6, GPIO.HIGH)
    GPIO.output(5, GPIO.LOW)
    time.sleep(1)
    print ('GPIO INITIALIZED')


def choose_action_using_simple_logic(distance_to_center):
    """ Naive policy to achive lane keeping, using simple rules.
    """
    if np.abs(distance_to_center) <= distance_threshold:
        # action = go_straight
        chosen_action_number = 0
    else:
        if distance_to_center < 0:
            # action = turn_left
            chosen_action_number = 3
        else:
            # action = turn_right
            chosen_action_number = 6
    return chosen_action_number


def feature_pre(s):
    """ The kernel function of state `s`.
    """
    s = s[np.newaxis,:]
    feature_sub = np.hstack((np.eye(1), s, s**2,[s[:,0]*s[:,1]])).transpose()
    return feature_sub


def choose_action_from_policy_w(x):
    q_values = np.matmul(w_matrix, x)
    chosen_action_number = np.argmax(q_values)
    return chosen_action_number


def main():
    try:
        with picamera.PiCamera(resolution='VGA') as camera:
            with io.BytesIO() as stream:
                    for frame in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
                        kpi_time = time.time()
                        stream.seek(0)
                        image = Image.open(stream)
                        img = np.array(image)
                        key = cv2.waitKey(1) & 0xFF #necessary for cv2

                        if (firstStart == True):
                            pwmR.ChangeDutyCycle(60)
                            pwmL.ChangeDutyCycle(50)
                            firstStart = False

                        # cropping image
                        img_cropped = crop_image(img, 0.45, 0.85)
                        # downsampling cropped image
                        img_downsampled = cv2.resize(img_cropped, size_after_downsampling, interpolation=cv2.INTER_LINEAR)

                        # color filtering image
                        laneIMG = lane_filter(img_downsampled, lower_lane_color1, upper_lane_color1)
                        # making image to a binary representation
                        laneIMG_binary = laneIMG/255

                        ## find peaks as the starting points of the lanes (left and right)
                        lane_center_left, lane_center_right = find_lane_centers(laneIMG_binary, left_peak_previous, right_peak_previous)

                        if (lane_center_left == False and lane_center_right== False):
                            print('End of the trial: No lines')
                            pwmR.stop()
                            pwmL.stop()
                            GPIO.cleanup()
                            np.save('memory_', memory)
                            break
                        elif (lane_center_left == False):
                            w_left = w_left_previous
                            # print('left line is missing')
                            # pwmR.stop()
                            # pwmL.stop()
                            # GPIO.cleanup()
                            # np.save('memory_', memory)
                            # break

                            # polynmial fitting for right lane
                            x_right, y_right = find_pixels_of_lane(laneIMG_binary, lane_center_right, window_size, width_of_laneIMG_binary)
                            try:
                                w_right = np.polyfit(x_right, y_right, poly_order)
                            except ValueError:
                                w_right = w_right_previous
                                # print('ValueError-00! Stop!')
                                # pwmR.stop()
                                # pwmL.stop()
                                # GPIO.cleanup()
                                # break
                            except np.RankWarning:
                                print('Rank Warning!!!')
                                w_right = w_right_previous


                        elif (lane_center_right== False):
                            w_right = w_right_previous
                            # print('right line is missing')
                            # pwmR.stop()
                            # pwmL.stop()
                            # GPIO.cleanup()
                            # np.save('memory_', memory)
                            # break

                            # polynmial fitting for left lane
                            x_left, y_left = find_pixels_of_lane(laneIMG_binary, lane_center_left, window_size, width_of_laneIMG_binary)
                            try:
                                w_left = np.polyfit(x_left, y_left, poly_order)
                            except ValueError:
                                w_left = w_left_previous
                                # print('ValueError-01! Stop!')
                                # pwmR.stop()
                                # pwmL.stop()
                                # GPIO.cleanup()
                                # break
                            except np.RankWarning:
                                print('Rank Warning!!!')
                                w_left = w_left_previous
                                # pwmR.stop()
                                # pwmL.stop()
                                # GPIO.cleanup()
                                # break
                        else:
                            left_peak_previous, right_peak_previous = lane_center_left, lane_center_right

                            # use a window to find all pixels of the left
                            # lane and the right lane

                            # polynmial fitting for left lane
                            x_left, y_left = find_pixels_of_lane(laneIMG_binary, lane_center_left, window_size, width_of_laneIMG_binary)
                            try:
                                w_left = np.polyfit(x_left, y_left, poly_order)
                            except ValueError:
                                w_left = w_left_previous
##                                print('ValueError-02! Stop!')
##                                pwmR.stop()
##                                pwmL.stop()
##                                GPIO.cleanup()
##                                break
                            except np.RankWarning:
                                print('Rank Warning!!!')
                                w_left = w_left_previous

                            # polynmial fitting for right lane
                            x_right, y_right = find_pixels_of_lane(laneIMG_binary, lane_center_right, window_size, width_of_laneIMG_binary)
                            try:
                                w_right = np.polyfit(x_right, y_right, poly_order)
                            except ValueError:
                                w_right = w_right_previous
##                                print('ValueError-03! Stop!')
##                                pwmR.stop()
##                                pwmL.stop()
##                                GPIO.cleanup()
##                                break
                            except np.RankWarning:
                                print('Rank Warning!!!')
                                w_right = w_right_previous



                        # It is convenient to use poly1d objects
                        # for dealing with polynomials
                        poly_fit_left = np.poly1d(w_left)
                        y_left_fitted = poly_fit_left(x_fitted)
                        poly_fit_right = np.poly1d(w_right)
                        y_right_fitted = poly_fit_right(x_fitted)

                        # plot the lane centerline
                        w_mid = (w_left+w_right)/2
                        poly_fit_mid = np.poly1d(w_mid)
                        y_mid_fitted = poly_fit_mid(x_fitted)

                        # to store the previous curves for left and right lane
                        w_left_previous, w_right_previous = w_left, w_right

                        # plot the bottom point of the lane centerline
                        x_bottom = np.int(x_fitted[-1])
                        y_bottom = np.int(y_mid_fitted[-1])



                        # plot for fitting
                        pts_left = np.array([y_left_fitted, x_fitted], np.int32).transpose()
                        pts_right = np.array([y_right_fitted, x_fitted], np.int32).transpose()
                        pts_mid = np.array([y_mid_fitted, x_fitted], np.int32).transpose()
                        pts_mid_bottom = np.array([y_bottom, x_bottom], np.int32).transpose()
                        image_center = np.int(width_of_laneIMG_binary/2)

                        # compute the pixel distance between the image center and
                        # the bottom point of the lane centerline
                        # the car on the right: negative;
                        # the car on the left: positive;
                        distance_to_center = y_bottom - image_center


                        print('distance_to_center:', distance_to_center)


                        # compute curvature at some point x
                        # now, point x is in the middle (from height) of the lane centerline
                        x_curv = np.int(x_fitted[int(number_points_for_poly_fit/2)])
                        y_curv = np.int(y_mid_fitted[int(number_points_for_poly_fit/2)])

                        # find the projected postion of the car at the mid
                        vehilce_projected_pos = (image_center, x_curv)
                        #print('vehilce_projected_pos', vehilce_projected_pos)
                        distance_at_mid = y_curv - image_center
                        print('distance_at_mid:', distance_at_mid)

                        if (firstStart == True):
                            pwmR.ChangeDutyCycle(30)
                            pwmL.ChangeDutyCycle(80)
                            firstStart = False
                            distance_to_center_last = distance_to_center
                            distance_at_mid_last = distance_at_mid
                            continue

##                        if (np.abs(distance_to_center-distance_to_center_last)>=30 or (np.abs(distance_to_center)>=100)):
                        if (np.abs(distance_to_center-distance_to_center_last) >= threshold_distance_error or np.abs(distance_at_mid-distance_at_mid_last)>= threshold_distance_error):

##                            print('distance_to_center_final',distance_to_center)
##                            print('large error in distance')
                            distance_to_center = distance_to_center_last
                            distance_at_mid = distance_at_mid_last
                            print('End of the trial: Large error in distance')
##                            pwmR.stop()
##                            pwmL.stop()
##                            np.save('memory_', memory)
##                            GPIO.cleanup()
##                            break

                        else:
                            distance_to_center_last = distance_to_center
                            distance_at_mid_last = distance_at_mid

                        if (np.abs(distance_to_center)>=70):
                            distance_to_center_last = distance_to_center
                            print('End of the trial: deviate from the lane')

##                            pwmR.stop()
##                            pwmL.stop()
##                            GPIO.cleanup()
##                            np.save('memory_', memory)
##                            break

                        cv2.polylines(img_downsampled, [pts_left], False, (0,255,255), 1)
                        cv2.polylines(img_downsampled, [pts_right], False, (0,255,255), 1)
                        cv2.polylines(img_downsampled, [pts_mid], False, (0,255,0), 1)


                        first_order_derivative_at_x = first_order_derivative_of_second_poly(w_mid, x_curv)
##                        print('derivative_at_x', first_order_derivative_at_x)
                        intercept = compute_intercept(first_order_derivative_at_x, x_curv, y_curv)
                        second_order_derivative_at_x = second_order_derivative_of_second_poly(w_mid, x_curv)
                        # we can use first_order_derivative_at_x and intercept
                        # to plot the tangent line at x_curv
                        x_curv_1, y_curv_1, x_curv_2, y_curv_2 = compute_points_on_tangent_line(first_order_derivative_at_x,intercept, x_curv, length_tangent_line)


                        # here, we officially compute curvature at some point x
                        # in order to differentiate between left turn and right turn
                        # here we use the signed curvature
                        # but, here the signed is opposite to the original definition: https://en.wikipedia.org/wiki/Curvature#Signed_curvature
                        # will fix in the future
##                        curvature_at_x =  np.abs(second_order_derivative_at_x) / (1+first_order_derivative_at_x**2)**(3./2)
                        curvature_at_x =  -second_order_derivative_at_x / (1+first_order_derivative_at_x**2)**(3./2)

##                        print('curvature_at_x', curvature_at_x)

                        # the center of the image indicates the position of the car
                        vehicle_pos = (image_center, height_of_laneIMG_binary-2)
                        cv2.circle(img_downsampled, center=vehicle_pos, radius=2, color=(255,0,255))

##                        state = np.array([distance_to_center, distance_at_mid, first_order_derivative_at_x, intercept, curvature_at_x])
##                            print('state.shape', state.shape)
                        state = np.array([distance_to_center, distance_at_mid,curvature_at_x])

                        #######################
                        K_mid = 6
                        differential_drive = -K_mid * distance_at_mid
##                        differential_drive = -K_mid*distance_at_mid + (np.random.rand(1)-0.5)*20

                        pwm_mid = 50
                        pwm_l_new = np.clip(pwm_mid - differential_drive/2, 0, 100.0)
                        pwm_r_new = np.clip(pwm_mid + differential_drive/2, 0, 100.0)

                        memory[memory_counter, :] = np.hstack([state, differential_drive])
                        memory_counter += 1

                        pwmL.ChangeDutyCycle(pwm_l_new)
                        pwmR.ChangeDutyCycle(pwm_r_new)
                        print('pwm_l_new', pwm_l_new, 'pwm_r_new', pwm_r_new)
                        ############################

                        # lane_center_bottom is the location where the fitted mid line crosses the lower boundary of the image
                        # we use (y_bottom, x_bottom) to indicate it
                        lane_center_bottom = (y_bottom, x_bottom-2)
                        cv2.circle(img_downsampled, center=lane_center_bottom, radius=2, color=(255,0,0))

                        # plot a circle to indicate the projected vehicle position at mid
                        cv2.circle(img_downsampled, center=vehilce_projected_pos, radius=2, color=(255,0,255))


                        # this plots a circle for (y_curv, x_curv)
                        cv2.circle(img_downsampled, center=(y_curv, x_curv), radius=2, color=(255,0,0))

                        # plot tangent line at (x_curv, y_curv), from (y_curv_1,x_curv_1) to (y_curv_2,x_curv_2)
                        cv2.line(img_downsampled, (y_curv_1,x_curv_1), (y_curv_2,x_curv_2),(0,0,255), 1, lineType = 8)

                        # img_downsampled_zoomed = cv2.resize(img_downsampled, (0,0), fx=4, fy=4)
                        # cv2.imshow('img_downsampled_zoomed', img_downsampled_zoomed)

                        print(time.time() - kpi_time)
                        stream.seek(0)
                        stream.truncate() # remove all previous images

    except KeyboardInterrupt:
        pwmR.stop()
        pwmL.stop()
        np.save('memory_', memory)
        print('End of the trial: KeyboardInterrupt')
        GPIO.cleanup()

if __name__ == '__main__':
    main()
