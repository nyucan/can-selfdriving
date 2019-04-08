# python 2.7
from __future__ import absolute_import, division, print_function
import io
from math import floor
import numpy as np
from time import sleep, time
# from os import listdir

from os.path import join, isfile

from control.motor import Motor
from control.policy import Policy

# input_file_names = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]


class Controller(object):
    def __init__(self):
        self.motor = Motor(slient=True)
        self.policy = Policy()
        self._start_time = time()
        self.is_recording = False
        self.K_im_traj = np.load('./control/K_traj_IM_VI.npy')
        # self.K_coupled = np.load('./control/coupled_k/0221.npy')
        self.testing_Ks = [
            'controller_q11_100_q_22_10_q_33_05npy.npy',  # 0
            'controller_q11_10_q_22_100_q_33_50npy.npy',  # 1
            'controller_q11_100_q_22_01_q_33_05npy.npy',  # 2 best for now
            'controller_q11_10_q_22_100_q_33_05npy.npy',  # 3
            'controller_q11_10_q_22_10_q_33_005npy.npy',  # 4
            'controller_q11_10_q_22_100_q_33_005npy.npy', # 5
            'controller_q11_100_q_22_01_q_33_005npy.npy', # 6 bad
            'controller_q11_100_q_22_10_q_33_50npy.npy',  # 7 very bad
            'controller_q11_100_q_22_100_q_33_05npy.npy', # 8 very bad
            'controller_q11_01_q_22_100_q_33_005npy.npy', # 9 bad
            'controller_q11_01_q_22_10_q_33_005npy.npy',  # 10 very very bad
            'controller_q11_10_q_22_10_q_33_05npy.npy',   # 11
            'controller_q11_01_q_22_01_q_33_005npy.npy',  # 12 good
            'controller_q11_01_q_22_100_q_33_05npy.npy',  # 13
            'controller_q11_10_q_22_01_q_33_50npy.npy',   # 14
            'controller_q11_100_q_22_100_q_33_50npy.npy', # 15 good
            'controller_q11_01_q_22_10_q_33_05npy.npy',   # 16
            'controller_q11_01_q_22_10_q_33_50npy.npy',   # 17
            'controller_q11_01_q_22_100_q_33_50npy.npy',  # 18
            'controller_q11_01_q_22_01_q_33_05npy.npy',   # 19
            'controller_q11_100_q_22_100_q_33_005npy.npy',  # 20
            'controller_q11_10_q_22_01_q_33_05npy.npy',   # 21
            'controller_q11_01_q_22_01_q_33_50npy.npy',   # 22
            'controller_q11_10_q_22_01_q_33_005npy.npy',  # 23
            'controller_q11_100_q_22_01_q_33_50npy.npy',  # 24
            'controller_q11_100_q_22_10_q_33_005npy.npy', # 25
            'controller_q11_10_q_22_10_q_33_50npy.npy'
        ]
        self.K_coupled = np.load('./control/coupled_k/controllers-0221/' + self.testing_Ks[2])
        
        self.dis_sum = 0
        self.z = np.zeros((2))
        self.threshold = 500
        self.init_record()

    def init_record(self):
        self.is_recording = True
        self.counter = 1
        self.record = []

    def finish_control(self):
        print('contorller: stop')
        self.motor.motor_stop()

    def make_decision_with_policy(self, policy_type, *args):
        """ Make decision with different policies.
            @param policy_type
                1: ADP
                2: pure pursuit
                3: Car following with ADP
                5: Coupled Car Following Controller
        """
        if policy_type == 1:    # ADP
            assert len(args) == 2, 'args should be exactly 2'
            cur_K = -self.K_im_traj[-1]
            distance_2_tan, radian_at_tan = args
            self.dis_sum += distance_2_tan
            pwm_l_new, pwm_r_new = Policy.adp(distance_2_tan, radian_at_tan, self.dis_sum, cur_K)
        elif policy_type == 2:  # pure pursuit
            l_d, sin_alpha = args
            amp = 150
            pwm_l_new, pwm_r_new = Policy.pure_pursuit(l_d, sin_alpha, amp)
        elif policy_type == 3:  # Car following with ADP
            assert len(args) == 3, 'args should be exactly 3'
            cur_K = -self.K_im_traj[-1]
            distance_2_tan, radian_at_tan, estimated_dis = args
            self.dis_sum += distance_2_tan
            pwm_l_new, pwm_r_new = Policy.car_following_with_adp(distance_2_tan, radian_at_tan, self.dis_sum, cur_K, estimated_dis)
        elif policy_type == 4:
            pass
            # removed
            # K = 0.5
            # dis2car, = args
            # pwm_l_new, pwm_r_new = Policy.car_following(dis2car, K)
        elif policy_type == 5:
            if self.is_recording and self.counter % 100 == 0:
                np.save('./.out/record-3-5', self.record)
            d_arc, d_curve, theta = args
            pwm_l_new, pwm_r_new = self.policy.adp_coupled_car_following(d_arc, d_curve, theta, self.z, self.K_coupled, self.record)
            print('counter: ', self.counter)
            self.counter += 1
        else:
            pwm_l_new, pwm_r_new = 0, 0
            print('Policy Not Found')
        self.motor.motor_set_new_speed(pwm_l_new, pwm_r_new)

    def start(self):
        self.motor.motor_startup()

    def cleanup(self):
        self.motor.motor_cleanup()

    def collision_avoid(self, start_time):
        """ Hardcoded collision avoidance behavior.
        """
        while True:
            cur_time = time() - start_time
            if cur_time < 3:
                self.motor.motor_set_new_speed(100, 9)
                print('ob2', time() - start_time)
            elif cur_time < 5.7:
                self.motor.motor_set_new_speed(15, 98)
                print('ob3', time() - start_time)
            elif cur_time < 6.3:
                self.motor.motor_set_new_speed(100, 20)
                print('ob4', time() - start_time)  
            else:
                print('obchange', time() - start_time)
                break
