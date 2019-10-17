import numpy as np


class OU_noise(object):
    def __init__(self, num_actions, action_low_bound, action_high_bound, dt,
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
        self.reset()

    def reset(self):
        self.state = np.zeros(self.num_actions)

    # self.state = np.zeros(self.num_actions)
    def state_update(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.num_actions)  # np.random.randn()生成0,1的随机数
        self.state = x + dx

    def add_noise(self, action):
        self.state_update()
        state = self.state
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, self.dt)
        return np.clip(action + state, self.action_low, self.action_high)
