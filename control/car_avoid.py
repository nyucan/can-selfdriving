# python 2.7
from __future__ import absolute_import, division, print_function
import io
from math import floor
import numpy as np
import cv2
from time import sleep, time
from os.path import join

from control.motor import Motor

class CarAvoid(object):
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

    def collision_avoid(self, start_time):
        ob_op = True
        while ob_op:
            # if time() - start_time < 1:
            #     self.motor.motor_set_new_speed(60,45)
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
                ob_op =False
                print('obchange', time() - start_time)

    # pass
