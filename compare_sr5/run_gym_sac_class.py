# 导入一些其他的必要包
import numpy as np
import time
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
import sys

sys.path.append("../")
# 选择强化算法
from sac_sp.SAC_class import SAC

# 导入log包!
from sp_utils.logx import EpochLogger
from sp_utils.logx import setup_logger_kwargs

# 选择环境
import gym
# import robosuite as suite

set_cameras = ['front', 'bird', 'agent', 'right']
Image_size = 256


def test_agent(args, net, env, n=5, logger=None):
    ep_reward_list = []
    for j in range(n):
        obs = env.reset()
        ep_reward = 0
        for i in range(args.max_steps):
            # Take deterministic actions at test time (noise_scale=0)
            s = np.hstack((obs['gripper_site_pos'],
                                    obs['joint_pos'],
                                    obs['robot-state'],
                                    obs["object-state"],
                                    ))

            a = net.get_action(s)
            obs, r_total, d, _ = env.step(a)
            r, gripper_flag, step_done_flag = r_total

            ep_reward += r
            if logger:
                logger.store(TestEpRet=ep_reward)

        ep_reward_list.append(ep_reward)
    mean_ep_reward = np.mean(np.array(ep_reward_list))
    if logger:
        return mean_ep_reward, logger
    else:
        return mean_ep_reward


def main():

    # 确定随机种子
    random_seed = int(time.time() * 10000 % 10000)
    # 设置传参和默认值
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--noise_scale', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=random_seed)
    # 默认的epochs=5000!
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--max_steps', type=int, default=200)
    # 实验名字需要改对应起来
    parser.add_argument('--exp_name', type=str, default='sac_sr5')

    args = parser.parse_args()

    tf.reset_default_graph()

    # 实例化log函数!
    logger_kwargs = setup_logger_kwargs(exp_name=args.exp_name,
                                        seed=args.seed,
                                        output_dir="../sp_data_logs/")
    logger = EpochLogger(**logger_kwargs)

    print("locals():", locals())
    logger.save_config(locals())

    # 创建虚拟环境
    camera_name = []
    for i in range(len(set_cameras)):
        camera_name.append(set_cameras[i] + "view")
    env = gym.make(args.env)
    # 设置环境的随机种子:robosuite可能没有
    # env.seed(args.seed)
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    obs = env.reset()
    perception_dim = env.observation_space.shape[0]

    # 确定state和action维度和动作上限
    s_dim = perception_dim
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high[0]

    # 创建强化算法类,里面还有一些参数,需要看里面的代码
    # SAC主要调整alpha,从0.1到0.25,找到最佳的一组
    net = SAC(a_dim, s_dim, a_bound,
              alpha=args.alpha,
              batch_size=args.batch_size,
              )

    # 设定保存的一些参数.
    ep_reward_list = []
    test_ep_reward_list = []
    start_time = time.time()
    # 主循环
    for i in range(args.epochs):
        # 环境的重置和一些变量的归零
        obs = env.reset()
        s = obs
        ep_reward = 0
        episode_time = time.time()
        for j in range(args.max_steps):
            # 选择动作
            # Add exploration noise
            a = net.get_action(s, args.noise_scale)

            a = np.clip(a, -a_bound, a_bound)

            obs, r, done, info = env.step(a)

            s_ = obs
            net.store_transition((s, a, r, s_, done))

            s = s_
            ep_reward += r
            if j == args.max_steps - 1:
                # 存episode reward.
                logger.store(EpRet=ep_reward)
                for _ in range(args.max_steps):
                    net.learn()

                ep_reward_list.append(ep_reward)
                print('Episode:', i, ' Reward: %0.4f' % float(ep_reward),
                      "learn step:", net.learn_step)

                # 增加测试部分!
                if i % 20 == 0:
                    test_ep_reward, logger = test_agent(args=args,
                                                        net=net,
                                                        env=env,
                                                        n=5,
                                                        logger=logger
                                                        )
                    test_ep_reward_list.append(test_ep_reward)

                    logger.log_tabular('Epoch', i)
                    logger.log_tabular('EpRet', with_min_and_max=True)
                    logger.log_tabular('TestEpRet', with_min_and_max=True)
                    logger.log_tabular('TotalEnvInteracts', i*args.max_steps+j)
                    logger.log_tabular('TotalTime', time.time() - start_time)
                    # logger.log_tabular('EpisopeTime', time.time() - episode_time)
                    logger.dump_tabular()

                break


if __name__ == '__main__':

    main()
