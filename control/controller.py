# python 2.7
from __future__ import absolute_import, division, print_function
import io
from math import floor
import numpy as np
from time import sleep, time
from os.path import join

from control.motor import Motor


class Controller(object):
    def __init__(self):
        self.motor = Motor(slient=True)
        self._start_time = time()
        self.is_recording = False
        self.K_im_traj = np.load('./control/K_traj_IM_VI.npy')
        self.dis_sum = 0
        self.threshold = 500
        # self.init_record()

    def init_record(self):
        self.is_recording = True
        self.counter = 1
        self.record = []

    def finish_control(self):
        print('contorller: stop')
        self.motor.motor_stop()
        if self.is_recording:
            np.save(join('.', 'record', 'record'), np.array(self.record))

    def make_decision(self, distance_2_tan, radian_at_tan):
        """ Make decision with a list of parameters.
            @paras
                distance_2_tan
                radian_at_tan
        """
        cur_k_index = -1
        self.cur_K = -self.K_im_traj[cur_k_index]
        self.dis_sum += distance_2_tan
        state = np.array([distance_2_tan, radian_at_tan, self.dis_sum])
        differential_drive = np.clip(-np.matmul(self.cur_K, state), -100.0, 100.0)
        print('controller with k ' + str(cur_k_index) + ':', distance_2_tan, radian_at_tan)
        pwm_mid = 50.0
        pwm_l_new = pwm_mid - differential_drive / 2
        pwm_r_new = pwm_mid + differential_drive / 2
        self.motor.motor_set_new_speed(pwm_l_new, pwm_r_new)
        # -------- recording data --------
        if self.is_recording:
            self.counter += 1
            self.record.append((distance_2_tan, radian_at_tan, self.dis_sum, differential_drive))
            # check point
            if self.counter % 100 == 0:
                np.save(join('.', 'record', 'record'), np.array(self.record))
            self.counter += 1
        # --------------------------------
        print(time() - self._start_time)

    def start(self):
        self.motor.motor_startup()

    def cleanup(self):
        self.motor.motor_cleanup()

    def collision_avoid(self, start_time):
        """ Hardcoded collision avoidance behavior.
        """
        ob_op = True
        while ob_op:
            # if time() - start_time < 1:
            #     self.motor.motor_set_new_speed(60, 45)
            #     print('ob1', time() - start_time)
            if time() - start_time < 3:
                self.motor.motor_set_new_speed(100, 9)
                print('ob2', time() - start_time)
            elif time() - start_time < 5.7:
                self.motor.motor_set_new_speed(15, 98)
                print('ob3', time() - start_time)
            elif time() - start_time < 6.3:
                self.motor.motor_set_new_speed(100, 20)
                print('ob4', time() - start_time)  
            else:
                # self.motor.motor_set_new_speed(pwm_l_new, pwm_r_new)
                ob_op = False
                print('obchange', time() - start_time)
