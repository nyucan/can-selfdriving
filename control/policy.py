import numpy as np

def adp(distance_2_tan, radian_at_tan, distance_integral, K):
    """ Control with `distance_2_tan`, `radian_at_tan` and `distance_integral`
        with `K` trained from the ADP algorithm.
    """
    state = np.array([distance_2_tan, radian_at_tan, distance_integral])
    differential_drive = np.clip(-np.matmul(K, state), -100.0, 100.0)
    print('ADP controller:', distance_2_tan, radian_at_tan)
    pwm_mid = 50
    pwm_l_new = np.clip(pwm_mid - differential_drive / 2, 0, 100)
    pwm_r_new = np.clip(pwm_mid + differential_drive / 2, 0, 100)
    return pwm_l_new, pwm_r_new


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


def car_following_with_adp(distance_2_tan, radian_at_tan, distance_2_car, distance_integral, K, dis_K):
    """ Control with `distance_2_tan`, `radian_at_tan` and `distance_integral`
        with `K` trained from the ADP algorithm.
        While following the car in front of it with a simple P controller and `distance_2_car`.
    """
    state = np.array([distance_2_tan, radian_at_tan, distance_integral])
    if distance_2_car < 45:
        return 0, 0
    elif distance_2_car >= 100:
        pwm_mid = 50
    else:
        pwm_mid = np.clip(45.0 + dis_K * distance_2_car, 40, 50)
    differential_drive = np.clip(-np.matmul(K, state), - 2 * pwm_mid, 2 * pwm_mid)
    print('ADP controller:', distance_2_tan, radian_at_tan, pwm_mid, differential_drive)
    pwm_l_new = np.clip(pwm_mid - differential_drive // 2, 0, 100)
    pwm_r_new = np.clip(pwm_mid + differential_drive // 2, 0, 100)
    return pwm_l_new, pwm_r_new
