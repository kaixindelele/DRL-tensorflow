import numpy as np


class Simple_noise(object):
    def __init__(self, num_actions, action_low_bound, action_high_bound,
                 dt=0.0001,
                 mu=0.0, theta=0.15, max_sigma=2.0, min_sigma=0.1):
        self.mu = mu  # 0.0
        self.theta = theta  # 0.15
        self.sigma = max_sigma  # 0.3
        self.max_sigma = max_sigma  # 0.3
        self.min_sigma = min_sigma  # 0.1
        self.dt = dt  # 0.001
        self.num_actions = num_actions  # 1
        self.action_low = action_low_bound  # -2
        self.action_high = action_high_bound  # 2

    def add_noise(self, action):
        action += self.max_sigma * np.random.randn(self.num_actions)
        return np.clip(action, self.action_low, self.action_high)
