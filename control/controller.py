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
        self.init_record()
        self.K_im_traj = np.load('./control/K_traj_IM_VI.npy')
        self.dis_sum = 0
        self.threshold = 500

    def init_record(self):
        self.counter = 1
        self.record = []
        # record_size = 5000
        # self.dis_record = np.zeros((record_size, 1))

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
        self.motor.motor_stop()
        # np.save(join('.', 'record', 'memory_'), self.memory)
        np.save(join('.', 'record', 'record'), np.array(self.record))

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

    def make_decision(self, distance_2_tan, radian_at_tan, start_time):
        """ Make decision with a list of parameters.
            @paras
                distance_2_tan
                radian_at_tan
        """
        # cur_k_index = int((c) / 20) * 1 + 3
        # self.counter = self.counter+1
        self.counter += 1
        cur_k_index = -1
        self.cur_K = -self.K_im_traj[cur_k_index]
        
        # if abs(self.dis_sum + distance_2_tan) < self.threshold:
        self.dis_sum += distance_2_tan
        state = np.array([distance_2_tan, radian_at_tan, self.dis_sum])
        differential_drive = np.clip(-np.matmul(self.cur_K, state), -100.0, 100.0)
        # self.memory[self.memory_counter, :] = np.hstack([state, differential_drive])
        print('controller with k ' + str(cur_k_index) + ':', distance_2_tan, radian_at_tan)
        # self.memory_counter += 1
        pwm_mid = 50.0
        pwm_l_new = pwm_mid - differential_drive / 2
        pwm_r_new = pwm_mid + differential_drive / 2
        # self.motor.motor_set_new_speed(pwm_l_new, pwm_r_new)
        if time() - start_time < 2:
            self.motor.motor_set_new_speed(60,45)
            print('1')
        elif time() - start_time < 5:
            self.motor.motor_set_new_speed(100, 25)
            print('2')
        elif time() - start_time < 8:
            self.motor.motor_set_new_speed(40,80)
            print('3')
        elif time() - start_time < 10.5:
            self.motor.motor_set_new_speed(100,25)
            print('4')
        else:
            self.motor.motor_set_new_speed(60,45)
            print('5')
        # self.record.append((distance_2_tan, radian_at_tan, self.dis_sum, differential_drive))
        # check point
        # if self.counter % 100 == 0:
        #     np.save(join('.', 'record', 'record'), np.array(self.record))
        # self.counter += 1
        print(time() - self._start_time)

    def start(self):
        self.motor.motor_startup()

    def cleanup(self):
        self.motor.motor_cleanup()
