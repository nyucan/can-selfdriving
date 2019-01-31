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
