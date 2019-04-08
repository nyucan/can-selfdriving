import numpy as np
from control.filters import DistanceKalmanFilter

class Policy(object):
    def __init__(self):
        self.last_pwd_mid = 60
        self.filter = DistanceKalmanFilter()

    @staticmethod
    def adp(distance_2_tan, radian_at_tan, distance_integral, K):
        """ Control with `distance_2_tan`, `radian_at_tan` and `distance_integral`
            with `K` trained from the ADP algorithm.
        """
        state = np.array([distance_2_tan, radian_at_tan, distance_integral])
        differential_drive = np.clip(-np.matmul(K, state), -100.0, 100.0)
        print('ADP controller:', distance_2_tan, radian_at_tan)
        pwm_mid = 60
        pwm_l_new = np.clip(pwm_mid - differential_drive / 2, 0, 100)
        pwm_r_new = np.clip(pwm_mid + differential_drive / 2, 0, 100)
        return pwm_l_new, pwm_r_new

    @staticmethod
    def pure_pursuit(l_d, sin_alpha, amp):
        """ Control with pure persuit method
            L: distance between two wheel
            sin_alpha: angle error
            l_d: distance from the vehicle to the center point
        """
        L = 32
        delta = np.arctan(2*L*sin_alpha/l_d)
        pwm_mid = 50.0
        pwm_l_new = np.clip(pwm_mid - amp * delta, 0, 100.0)
        pwm_r_new = np.clip(pwm_mid + amp * delta, 0, 100.0)
        print(delta, l_d, sin_alpha, pwm_l_new, pwm_r_new)
        return pwm_l_new, pwm_r_new

    @staticmethod
    def car_following_with_adp(distance_2_tan, radian_at_tan, distance_integral, K, estimated_dis):
        """ Control with `distance_2_tan`, `radian_at_tan` and `distance_integral`
            with `K` trained from the ADP algorithm.
            While following the car in front of it with a simple P controller and `distance_2_car`.
        """
        state = np.array([distance_2_tan, radian_at_tan, distance_integral])
        MID_K = 1.5
        diff = estimated_dis - 70  # try to stay 70cm away from the previous car
        pwm_mid = 60
        if diff < -40:
            return 0, 0
        elif diff >= 60:
            pwm_mid = 60
        else:
            pwm_mid = np.clip(45.0 + MID_K * diff, 30, 60)
        print('distance:', estimated_dis, 'diff:', diff, 'mid:', pwm_mid)
        # rec.append([estimated_dis, pwm_mid, distance_2_tan, radian_at_tan, distance_integral])
        differential_drive = np.clip(-np.matmul(K, state), -100.0, 100.0)
        pwm_l_new = np.clip(pwm_mid - differential_drive / 2, 0, 100)
        pwm_r_new = np.clip(pwm_mid + differential_drive / 2, 0, 100)
        return pwm_l_new, pwm_r_new

    # @staticmethod
    # def car_following(distance_2_car, K):
    #     pwm_mid = 50
    #     diff_distance = distance_2_car - 70
    #     if diff_distance < -50:
    #         return 0, 0
    #     elif diff_distance >= 50:
    #         pwm_mid = 50
    #     else:
    #         pwm_mid = np.clip(pwm_mid + K * diff_distance, 0, 100)
    #     pwm_l_new, pwm_r_new = pwm_mid, pwm_mid
    #     print(distance_2_car, diff_distance, pwm_mid)
    #     # return pwm_l_new, pwm_r_new + 5   # for car 2
    #     # return pwm_l_new + 10, pwm_r_new  # cor car 1


    def adp_coupled_car_following(self, d_arc, d_curve, theta, z, K, rec):
        """ A coupled controller.
        """
        d_arc_origin = d_arc
        d_arc = self.filter.apply(d_arc, self.last_pwd_mid)
        print('distance:', d_arc_origin, d_arc)
        state = np.array([d_arc, d_curve, theta])
        z += np.array([d_arc - 70, d_curve])
        state_aug = np.concatenate((state, z))
        u = -K.dot(state_aug)
        u[1] = -u[1]
        pwm_mid = np.clip(u[0], 30, 60)
        self.last_pwd_mid = pwm_mid
        pwm_l_new = np.clip(pwm_mid - u[1] / 2, 0, 100)
        pwm_r_new = np.clip(pwm_mid + u[1] / 2, 0, 100)
        record_item = np.concatenate((state, [pwm_mid, u[1], d_arc_origin]))
        rec.append(record_item)
        return pwm_l_new, pwm_r_new

    def no_orientation_control(self, d_curve, theta, l_d):
        """Path following with no orientation control in Handbook of Robotics P805
        """
        u_1 = 50
        l_1 = l_d 
        K = 2
        u_2 = -(u_1/l_1)*(np.sin(theta)/np.cos(theta)) - (u_1/np.cos(theta))*K*d_curve
        print('u_2:', u_2, 'd_curve:', d_curve, 'theta:', theta)
        pwm_mid = 50.0
        pwm_l_new = np.clip(pwm_mid - u_2, 0, 100.0)
        pwm_r_new = np.clip(pwm_mid + u_2, 0, 100.0)
        return  pwm_l_new, pwm_r_new
