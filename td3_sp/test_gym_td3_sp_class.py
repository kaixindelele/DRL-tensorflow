import matplotlib.pyplot as plt
import numpy as np
import gym
import time
import sys

sys.path.append("../")
from td3_sp.TD3_class import TD3

MAX_EPISODES = 250
MAX_EP_STEPS = 1000

RENDER = False
ENV_NAME = 'Hopper-v2'


def test_agent(net, env, n=10):
    ep_reward_list = []
    for j in range(n):
        s = env.reset()
        ep_reward = 0
        for i in range(MAX_EP_STEPS):
            # Take deterministic actions at test time (noise_scale=0)
            s, r, d, _ = env.step(net.get_action(s))
            ep_reward += r

        ep_reward_list.append(ep_reward)
    mean_ep_reward = np.mean(np.array(ep_reward_list))
    return mean_ep_reward


def main():

    env = gym.make(ENV_NAME)
    env = env.unwrapped
	seed=4
    env.seed(seed)
	tf.set_random_seed(seed=seed)
    np.random.seed(seed=seed)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high[0]

    net = TD3(a_dim, s_dim, a_bound,
              batch_size=100,
              )
    ep_reward_list = []
    test_ep_reward_list = []
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # Add exploration noise
            if i < 10:
                a = np.random.rand(a_dim) * a_bound
            else:
                # a = net.choose_action(s)
                a = net.get_action(s, 0.1)
            # a = noise.add_noise(a)

            a = np.clip(a, -a_bound, a_bound)

            s_, r, done, info = env.step(a)
            done = False if j == MAX_EP_STEPS-1 else done

            net.store_transition((s, a, r, s_, done))

            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS - 1:

                for _ in range(MAX_EP_STEPS):
                    net.learn()

                ep_reward_list.append(ep_reward)
                print('Episode:', i, ' Reward: %i' % int(ep_reward),
                      # 'Explore: %.2f' % var,
                      "learn step:", net.learn_step)
                # if ep_reward > -300:RENDER = True

                # 增加测试部分!
                if i % 20 == 0:
                    test_ep_reward = test_agent(net=net, env=env, n=5)
                    test_ep_reward_list.append(test_ep_reward)
                    print("-"*20)
                    print('Episode:', i, ' Reward: %i' % int(ep_reward),
                          'Test Reward: %i' % int(test_ep_reward),
                          )
                    print("-" * 20)

                break

    plt.plot(ep_reward_list)
    plt.show()
    plt.plot(test_ep_reward_list)
    plt.show()


if __name__ == '__main__':
    main()

