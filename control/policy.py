import numpy as np

def adp(distance_2_tan, radian_at_tan, distance_integral, K):
    """ Control with `distance_2_tan`, `radian_at_tan` and `distance_integral`
        with `K` trained from the ADP algorithm.
    """
    state = np.array([distance_2_tan, radian_at_tan, distance_integral])
    differential_drive = np.clip(-np.matmul(K, state), -100.0, 100.0)
    print('ADP controller:', distance_2_tan, radian_at_tan)
    pwm_mid = 55
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


def car_following_with_adp(distance_2_tan, radian_at_tan, distance_integral, K, estimated_dis, rec):
    """ Control with `distance_2_tan`, `radian_at_tan` and `distance_integral`
        with `K` trained from the ADP algorithm.
        While following the car in front of it with a simple P controller and `distance_2_car`.
    """
    state = np.array([distance_2_tan, radian_at_tan, distance_integral])
    MID_K = 0.1
    diff = estimated_dis - 70  # try to stay 70cm away from the previous car
    pwm_mid = 60
    if diff < -40:
        return 0, 0
    elif diff >= 60:
        pwm_mid = 60
    else:
        pwm_mid = np.clip(45.0 + MID_K * diff, 30, 60)
    print('distance:', estimated_dis, 'diff:', diff, 'mid:', pwm_mid)
    rec.append([estimated_dis, pwm_mid, distance_2_tan, radian_at_tan, distance_integral])
    differential_drive = np.clip(-np.matmul(K, state), -100.0, 100.0)
    pwm_l_new = np.clip(pwm_mid - differential_drive / 2, 0, 100)
    pwm_r_new = np.clip(pwm_mid + differential_drive / 2, 0, 100)
    return pwm_l_new, pwm_r_new


def car_following(distance_2_car, K):
    pwm_mid = 50
    diff_distance = distance_2_car - 70
    if diff_distance < -50:
        return 0, 0
    elif diff_distance >= 50:
        pwm_mid = 50
    else:
        pwm_mid = np.clip(pwm_mid + K * diff_distance, 0, 100)
    pwm_l_new, pwm_r_new = pwm_mid, pwm_mid
    print(distance_2_car, diff_distance, pwm_mid)
    # return pwm_l_new, pwm_r_new + 5   # for car 2
    # return pwm_l_new + 10, pwm_r_new  # cor car 1
