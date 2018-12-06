# python 2.7
from __future__ import absolute_import, division, print_function
import io
from math import floor
import numpy as np
import cv2
from time import sleep, time
from os.path import join

from control.motor import Motor


class Controller(object):
    def __init__(self):
        self.motor = Motor(slient=True)
        self._start_time = time()
        # self.init_memory()
        # self.init_record()
        self.K_im_traj = np.load('./control/K_traj_IM_VI.npy')
        self.dis_sum = 0
        self.threshold = 500

    def init_record(self):
        self.counter = 1
        self.record = []

    def init_memory(self):
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

        # suppose we have 2 states to store: distance_2_tan, angle_of_tan
        dim_state = 2

        # threshold_distance_error determines if the distances_to_center is corrupted/wrongly measured
        self.threshold_distance_error = 50

        # memory for storing states and actions
        # memory_size = 5000
        # self.memory_counter = 0
        # self.memory = np.zeros((memory_size, dim_state + 1))

    def finish_control(self):
        print('contorller: stop')
        self.basespeed = 40
        # self.motor.motor_stop()
        # np.save(join('.', 'record', 'memory_'), self.memory)
        # np.save(join('.', 'record', 'record'), np.array(self.record))

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

    def make_decision(self, type, *args):
        """ Make decision, the overloaded version.
        """
        if type == 'adp_lane_keeping_decision':
            self.adp_lane_keeping_decision(args[0], args[1])
        elif type == 'p_lane_keeping_decision':
            pass
        elif type == 'manual_follow_decision':
            self.manual_follow_decision(args[0], args[1], args[2])

    def adp_lane_keeping_decision(self, distance_2_tan, radian_at_tan):
        ## car 1
        self.basespeed = 50
        ## car 2
        # self.basespeed = 55
        cur_k_index = -1
        self.cur_K = -self.K_im_traj[cur_k_index]
        self.dis_sum += distance_2_tan
        state = np.array([distance_2_tan, radian_at_tan, self.dis_sum])
        differential_drive = -np.matmul(self.cur_K, state)
        pwm_l_new = np.clip(self.basespeed - differential_drive / 2, 0, 100)
        pwm_r_new = np.clip(self.basespeed + differential_drive / 2, 0, 100)
        self.motor.motor_set_new_speed(pwm_l_new, pwm_r_new)

    def manual_follow_decision(self, distance_2_tan, radian_at_tan, distance2car):
        k_speed = -1
        base_distance = 27
        
        # control the base speed
        if distance2car < 20:
            self.basespeed = 30
        elif distance2car < 25:
            self.basespeed = 40
        else:
            self.basespeed = k_speed * (base_distance - distance2car) + 50
            np.clip(self.basespeed, 42, 58)
        print('speed: ', self.basespeed)

        self.cur_K = -self.K_im_traj[-1]
        self.dis_sum += distance_2_tan
        state = np.array([distance_2_tan, radian_at_tan, self.dis_sum])
        differential_drive = -np.matmul(self.cur_K, state)
        pwm_l_new = np.clip(self.basespeed - differential_drive / 2, 0, 100)
        pwm_r_new = np.clip(self.basespeed + differential_drive / 2, 0, 100)
        # print('new_speed:', pwm_l_new, pwm_r_new)
        self.motor.motor_set_new_speed(pwm_l_new, pwm_r_new)

    def start(self):
        self.motor.motor_startup()

    def cleanup(self):
        self.motor.motor_cleanup()
