# python 2.7
from __future__ import absolute_import, division, print_function
from os.path import join
import io
import sys
import picamera
import numpy as np
import cv2
import RPi.GPIO as GPIO
from PIL import Image
from time import sleep
from os.path import join

from util.detect_peaks import detect_peaks
import util.math_support as ms


class Controller(object):
    def __init__(self):
        self.init_q_learning()
        self.init_gpio()

    def init_gpio(self):
        GPIO.setmode(GPIO.BCM)
        ENA, ENB = 26, 11
        IN1, IN2, IN3, IN4 = 19, 13, 6, 5
        sleep(1)

        #  Motor Pins
        GPIO.setup(ENA, GPIO.OUT) # ENA
        GPIO.setup(ENB, GPIO.OUT) # ENB
        GPIO.setup(IN1, GPIO.OUT) # IN1
        GPIO.setup(IN2, GPIO.OUT) # IN2
        GPIO.setup(IN3, GPIO.OUT) # IN3
        GPIO.setup(IN4, GPIO.OUT) # IN4

        # PWM pin and Frequency
        self.pwmR = GPIO.PWM(26, 100)
        self.pwmL = GPIO.PWM(11, 100)
        self.pwmR.start(0)
        self.pwmL.start(0)
        sleep(1)

        GPIO.output(19, GPIO.HIGH)
        GPIO.output(13, GPIO.LOW)
        GPIO.output(6, GPIO.HIGH)
        GPIO.output(5, GPIO.LOW)
        sleep(1)
        print ('GPIO INITIALIZED')

    def init_q_learning(self):
        go_straight_00 = np.array([50, 60])
        go_straight_01 = np.array([40, 50])
        go_straight_02 = np.array([60, 70])
        turn_left_00 = np.array([0, 70])
        turn_left_01 = np.array([0, 90])
        turn_left_02 = np.array([30, 90])
        turn_right_00 = np.array([90, 0])
        turn_right_01 = np.array([100, 0])
        turn_right_02 = np.array([100, 30])

        self.action_set = np.concatenate([
            [go_straight_00], [go_straight_01], [go_straight_02],
            [turn_left_00], [turn_left_01], [turn_left_02],
            [turn_right_00], [turn_right_01], [turn_right_02]
        ])
        self.dim_action = self.action_set.shape[0]

        # random actions exclude the ones used in simple logic function
        self.rand_action_options = np.array([1, 2, 4, 5, 7, 8])

        #  threshold in pixels
        self.distance_threshold = 3

        # epsilon greedy: epsilon% to choose random action
        self.epsilon_greedy = 0.

        # suppose we have 5 states to store: distance_to_center, distance_at_mid, first_order_derivative_at_x, intercept, curvature
        ##dim_state = 5
        self.dim_state = 3

        # threshold_distance_error determines if the distances_to_center is corrupted/wrongly measured
        self.threshold_distance_error = 50

        # memory for storing states and actions
        memory_size = 1000
        self.memory_counter = 0
        self.memory = np.zeros((memory_size, dim_state + 1))

    def motor_startup(self):
        self.pwmR.ChangeDutyCycle(60)
        self.pwmL.ChangeDutyCycle(50)

    def motor_stop(self):
        self.pwmR.stop()
        self.pwmL.stop()
        np.save(join('.', 'q-models', 'memory_'), self.memory)
        GPIO.cleanup()

    def motor_set_new_speed(self, left, right):
        self.pwmL.ChangeDutyCycle(left)
        self.pwmR.ChangeDutyCycle(right)
        print('Controller: ', 'pwm_l_new', pwm_l_new, 'pwm_r_new', pwm_r_new)

    def choose_action_using_simple_logic(self, distance_to_center):
        """ Naive policy to achive lane keeping, using simple rules.
        """
        if np.abs(distance_to_center) <= self.distance_threshold:
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

    def feature_pre(self, s):
        """ Kernel function of state `s`.
        """
        s = s[np.newaxis,:]
        feature_sub = np.hstack((np.eye(1), s, s**2,[s[:,0]*s[:,1]])).transpose()
        return feature_sub

    def make_decisiton(self, distance_to_center, distance_at_mid, curvature_at_x):
        state = np.array([distance_to_center, distance_at_mid, curvature_at_x])
        K_mid = 6
        differential_drive = -K_mid*distance_at_mid

        pwm_mid = 50
        pwm_l_new = np.clip(pwm_mid - differential_drive/2, 0, 100.0)
        pwm_r_new = np.clip(pwm_mid + differential_drive/2, 0, 100.0)
        self.memory[self.memory_counter, :] = np.hstack([state, differential_drive])
        self.memory_counter += 1

# load learned weights
## w = np.load('learned_weights_even_0725.npy')
## w_matrix = w.reshape((9,6))
# def choose_action_from_policy_w(x):
#     q_values = np.matmul(w_matrix, x)
#     chosen_action_number = np.argmax(q_values)
#     return chosen_action_number
