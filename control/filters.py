from collections import deque
from pykalman import KalmanFilter
import numpy as np

class Filter(object):
    def apply():
        pass


class LowPass(Filter):
    def __init__(self, num=5):
        self.q = deque()
        self.max_size = num
        self.s = 0

    def apply(self, newEle):
        if self.max_size == len(self.q):
            self.s -= self.q.popleft()
        self.s += newEle
        self.q.append(newEle)
        return self.s / len(self.q)


class DistanceKalmanFilter(object):
    def __init__(self):
        n_dim_state = 1
        self.current_state_means = np.array([90])
        self.current_state_cov =  np.zeros((n_dim_state, n_dim_state))
        self.kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            transition_covariance=[3],
            observation_covariance=[100],
            initial_state_mean=self.current_state_means,
            initial_state_covariance=self.current_state_cov
        )

    def apply(self, observed_distance, pwm_mid):
        self.current_state_means, self.current_state_cov = (
            self.kf.filter_update(
                self.current_state_means,
                self.current_state_cov,
                observed_distance,
                transition_offset=np.array([self.get_control_input(pwm_mid)])
            )
        )
        return self.current_state_means[0]

    def get_control_input(self, pwm):
        time_step = 0.08
        current_velocity = ((502.654 / 18.795) / 60) * pwm
        leading_velocity = (502.654 / 32.16)  # cm per sencod
        return (leading_velocity - current_velocity) * time_step
