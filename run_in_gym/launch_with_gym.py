import numpy as np
import tensorflow as tf
import gym
import time
import numpy as np
import tensorflow as tf
import gym
import os
import time
import sys

sys.path.append("../")


def run(seed=184,
        algo='td3', 
        per_flag=True,
        epochs=3000,
        gamma=0.99,
        RlNet=None,
        noise_size=0.1
        ):
    import argparse
    random_seed = seed
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=gamma)
    parser.add_argument('--seed', '-s', type=int, default=random_seed)
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument('--max_steps', type=int, default=1000)
    if per_flag:
        exp_name = algo+"_per"
    else:
        exp_name = algo
    parser.add_argument('--exp_name', type=str, default=exp_name)
    args = parser.parse_args()

    env = gym.make(args.env)
    env = env.unwrapped
    env.seed(args.seed)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high[0]

    
    net = RlNet(a_dim, s_dim, a_bound,
              gamma=gamma,              
              sess_opt=0.1,
              per_flag=per_flag
              )
    ep_reward_list = []
    test_ep_reward_list = []

    for i in range(args.epochs):
        s = env.reset()
        ep_reward = 0
        st = time.time()
        for j in range(args.max_steps):

            # Add exploration noise
            if i < 10:
                a = np.random.rand(a_dim) * a_bound
            else:
                a = net.get_action(s, noise_size)

            a = np.clip(a, -a_bound, a_bound)

            s_, r, done, info = env.step(a)
            done = False if j == args.max_steps - 1 else done

            net.store_transition((s, a, r, s_, done))

            s = s_
            ep_reward += r
            if j == args.max_steps - 1:
                up_st = time.time()
                for _ in range(args.max_steps):
                    net.learn()

                ep_update_time = time.time() - up_st

                ep_reward_list.append(ep_reward)
                print('Episode:', i, ' Reward: %i' % int(ep_reward),
                      # 'Explore: %.2f' % var,
                      "learn step:", net.learn_step,
                      "ep_time:", np.round(time.time()-st, 3),
                      "up_time:", np.round(ep_update_time, 3),
                      )
                # if ep_reward > -300:RENDER = True

                # 增加测试部分!
                if i % 20 == 0:
                    test_ep_reward = net.test_agent(env=env, n=5)
                    test_ep_reward_list.append(test_ep_reward)
                    print("-" * 20)
                    print('Episode:', i, ' Reward: %i' % int(ep_reward),
                          'Test Reward: %i' % int(test_ep_reward),
                          )
                    print("-" * 20)

                break

    import matplotlib.pyplot as plt

    plt.plot(ep_reward_list)
    img_name = str(args.exp_name + "_" + args.env + "_epochs" +
                   str(args.epochs) +
                   "_seed" + str(args.seed))
    plt.title(img_name + "_train")
    plt.savefig(img_name + ".png")
    plt.show()
    plt.close()

    plt.plot(test_ep_reward_list)
    plt.title(img_name + "_test")
    plt.savefig(img_name + ".png")
    plt.show()


if __name__ == '__main__':
    algo_index = 1
    seed = 184
    per_flag = True

    rl_algo_list = ["DDPG", "SAC_AUTO", "TD3", "SAC"]
    import rl_algorithms
    try:
        net = eval("rl_algorithms."+rl_algo_list[algo_index])
    except:        
        pass
    
    run(seed=seed,
        algo=rl_algo_list[algo_index], 
        per_flag=per_flag,
        epochs=3000,
        gamma=0.99,
        RlNet=net,
        noise_size=0.1)
