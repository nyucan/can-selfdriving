import numpy as np

def adp(distance_2_tan, radian_at_tan, distance_integral, K):
    """ Control with `distance_2_tan`, `radian_at_tan` and `distance_integral`
        with `K` trained from the ADP algorithm.
    """
    state = np.array([distance_2_tan, radian_at_tan, distance_integral])
    differential_drive = np.clip(-np.matmul(K, state), -100.0, 100.0)
    print('ADP controller:', distance_2_tan, radian_at_tan)
    pwm_mid = 50.0
    pwm_l_new = pwm_mid - differential_drive / 2
    pwm_r_new = pwm_mid + differential_drive / 2
    return pwm_l_new, pwm_r_new

def pure_pursuit(l_d, sin_alpha, amp):
    """Control with pure persuit method
       L: distance between two wheel
       sin_alpha: angle error
       l_d: distance from the vehicle to the center point
    """
    L = 32
    delta = np.arctan(2*L*sin_alpha/l_d)
    pwm_mid = 50.0
    pwm_l_new = np.clip(pwm_mid - amp * delta, 0.0, 100.0)
    pwm_r_new = np.clip(pwm_mid + amp * delta, 0.0, 100.0)
    print(delta, l_d, sin_alpha, pwm_l_new, pwm_r_new)
    return pwm_l_new, pwm_r_new

def VBC(Z):
    """Visual Based Control method
       Z: distance from center of Axes to the lane center point
    """

    pwm_mid = 50.0
    k = 3.0
    pwm_l_new = np.clip(pwm_mid + k * Z, 0.0, 100.0)
    pwm_r_new = np.clip(pwm_mid - k * Z, 0.0, 100.0)
    print(Z, pwm_l_new, pwm_r_new)
    return pwm_l_new, pwm_r_new
