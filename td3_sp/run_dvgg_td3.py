"""
就用这个作为最终的主程序!!!
调用net_per作为强化算法,元组为s,a,r,s_,done五个元素.
噪声每个episode做一次reset!
调用真正的回归奖励.

根据四卡机子上跑的最好的那组实验，想办法做一个复现！
1、用最好的epoch50做前端
2、奖励函数尽量一致！
3、设置一个随机种子
4、设计一个传参数
5、将真实奖励和预测奖励做一个记录和对比！
6、换回标准的夹爪方式（避免黑图）
7、将物块的位置限定在中心的6厘米范围内，和Robosuite默认的一样！！！

加满注释版,在单卡机器上不会黑图.
base on test_net_per_no_drop.py

by lyl
started 2019.09.3.21:53

"""
import sys

sys.path.append("../")
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import cv2

import os

try:
    device_id = sys.argv[5]
except:
    device_id = "0"
print("device_id:", device_id)
print("device_id:", type(device_id))

os.environ["CUDA_VISIBLE_DEVICES"] = device_id
import tensorflow as tf

# 加一个随机种子！
try:
    seed = int(sys.argv[2])
except:
    seed = 4
seed = int(time.time() * 100000 % 100000)
print("seed:", seed)
np.random.seed(seed)
tf.set_random_seed(seed)

try:
    MAX_EPISODES = int(sys.argv[4])
except:
    MAX_EPISODES = 5000
print("MAX_EPISODES:", MAX_EPISODES)
print("MAX_EPISODES:", type(MAX_EPISODES))

import robosuite as suite

from td3_sp.TD3_class import TD3
from noise.simple_noise import Simple_noise
from dvgg.dvgg_combine import DVGG

from master_utils.save_print_logs import Logger
from master_utils.transform import CameraTransform
from master_utils.main_utils import save_render_image, get_distances, get_shelters
from master_utils.check_cube import get_all_cube_rgb, get_all_shelter_info, check_on_table
from master_utils.plt_function import plt_function, plt_end_step
from master_utils.main_utils import get_new_reward, get_sparse_reward

set_cameras = ['front', 'bird', 'agent', 'right']
MAX_EP_STEPS = 200
Actor_lr = 0.001  # learning rate for actor
Critic_lr = 0.001  # learning rate for critic
Image_size = 256
Origin_size = 256

RENDER = False

LOAD = True
Shelter_flag = True
omit_drop_flag = False
# 是否需要将图片归一化,dvgg训练的是归一化的.
scale_flag = True
# 是否需要将图片翻转,dvgg训练的是翻转的
flip_flag = True
# 是否需要将图像转换颜色通道,dvgg训练的是标准的rgb通道.所以先不转换了
brg_flag = False
dvgg_epoch = 50
net_name = "VGG16"
# LOAD_dir = 'form-epoch' + str(dvgg_epoch) + '-batch_size-64-26axis-' + net_name
LOAD_dir = "finetune_dvgg_450_episode"
Save_end = 5

Train_step = 20
Test_step = 5

No_noise_number = int(MAX_EPISODES / 10)

ENV_NAME = 'Robosuite'


def test_agent(dvgg, net, env, n=10):
    ep_reward_list = []
    for j in range(n):
        obs = env.reset()
        ep_reward = 0
        for i in range(MAX_EP_STEPS):
            # Take deterministic actions at test time (noise_scale=0)
            perception = np.hstack((obs['gripper_site_pos'], obs['joint_pos'], obs['robot-state']))

            front_to_vgg, right_to_vgg = img_trans(obs)
            now_predict = dvgg.run_ones(front_to_vgg,
                                        right_to_vgg,
                                        )
            s = np.hstack((perception, now_predict))
            a = net.get_action(s)
            obs, r_total, d, _ = env.step(a)
            r, gripper_flag, step_done_flag = r_total

            ep_reward += r

        ep_reward_list.append(ep_reward)
    mean_ep_reward = np.mean(np.array(ep_reward_list))
    return mean_ep_reward


def img_trans(obs):
    if flip_flag:
        front = cv2.flip(obs["image"][0], 0)
        front = cv2.resize(front, (Image_size, Image_size))
        right = cv2.flip(obs["image"][3], 0)
        right = cv2.resize(right, (Image_size, Image_size))
    else:
        front = obs["image"][0]
        front = cv2.resize(front, (Image_size, Image_size))
        right = obs["image"][3]
        right = cv2.resize(right, (Image_size, Image_size))

    if brg_flag:
        front = cv2.cvtColor(front, cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

    # 在epoch99模型中没有归一化！
    if scale_flag:
        front_to_vgg = front / 255
        right_to_vgg = right / 255
    else:
        front_to_vgg = front
        right_to_vgg = right
    return front_to_vgg, right_to_vgg


def save_train_images2local(obs, i, j, logs_path, exp_names, save_end=Save_end):
    if i % 50 < save_end:
        front_path = logs_path + exp_names + '/images/episode_front_' + str(i) + "_step_" + str(
            j) + '_video.png'
        right_path = logs_path + exp_names + '/images/episode_right_' + str(i) + "_step_" + str(
            j) + '_video.png'
        if j < 10:
            if j % 2 == 0:
                save_render_image(front_path, obs["image"][0])
                save_render_image(right_path, obs["image"][3])
        elif j < 40:
            if j % 5 == 0:
                save_render_image(front_path, obs["image"][0])
                save_render_image(right_path, obs["image"][3])
        elif j > 170:
            if j % 2 == 0:
                save_render_image(front_path, obs["image"][0])
                save_render_image(right_path, obs["image"][3])
        elif j % 20 == 0:
            save_render_image(front_path, obs["image"][0])
            save_render_image(right_path, obs["image"][3])
        if j == 199:
            save_render_image(front_path, obs["image"][0])
            save_render_image(right_path, obs["image"][3])


def main(special_name="net_OUNoise_batch_size256", batch_size=128, memory_size=100000, sigma=0.15):
    tf.reset_default_graph()

    # 存训练部分的episode reward及其其他
    ep_r_list = []
    ep_reward_list = []
    # 存测试部分五次平均的episode reward及其其他
    test_ep_r_list = []
    mean_step_r_list = []
    plt_list = []
    test_plt_list = []
    end_step_r = []

    logs_path = '../clean_train_logs/'

    exp_names = ENV_NAME + "_" + special_name
    print(logs_path + exp_names)
    plt_save_path = logs_path + ENV_NAME + '_' + special_name + "/"

    try:
        os.mkdir(logs_path + exp_names)
        os.mkdir(logs_path + exp_names + '/images')
        os.mkdir(logs_path + exp_names + '/dvgg_data/')
        print("create a new file:", logs_path + exp_names)
    except Exception as e:
        print(e)
        print("the file has existed!")

    sys.stdout = Logger(logs_path + exp_names + "/print.log", sys.stdout)

    t1 = time.time()

    camera_name = []
    for i in range(len(set_cameras)):
        camera_name.append(set_cameras[i] + "view")

    env = suite.make(
        'PrintDisplay',
        has_renderer=True,
        use_camera_obs=True,
        camera_depth=False,
        ignore_done=False,
        render_visual_mesh=False,
        reward_shaping=True,
        camera_height=Origin_size,
        camera_width=Origin_size,
        camera_name=camera_name,
        control_freq=10,
        reach_flag=False,
    )
    obs = env.reset()
    perception_dim = obs['gripper_site_pos'].shape[0] + obs['robot-state'].shape[0] + obs["joint_pos"].shape[0]

    distances_dim = 24
    shelter_dim = 2
    if Shelter_flag:
        robot_state_dim = perception_dim + distances_dim + shelter_dim
    else:
        robot_state_dim = perception_dim + distances_dim
    a_dim = np.array(env.action_spec).shape[1]
    action_low_bound, action_high_bound = env.action_spec[0], env.action_spec[1]
    a_bound = action_high_bound

    # create some necessary class.
    # 这个output_dim是距离的dim,跟遮挡没有关系!
    axis_num = 21
    net_name = "VGG16"
    shelter_flag = Shelter_flag
    thin_flag = True
    scope_name = "DVGG_total"
    share_flag = False

    dvgg = DVGG(
        others_dim=axis_num,
        net_name=net_name,
        img_size=Image_size,
        shelter_flag=shelter_flag,
        thin_flag=thin_flag,
        scope_name=scope_name,
        share_flag=share_flag,
    )
    # dvgg参数加载路径
    dvgg_restore_path = '../clean_train_logs/' + LOAD_dir + "/param_logs"
    print("dvgg_restore_path:", dvgg_restore_path)
    # dvgg加载预训练参数
    dvgg.restore(dvgg_restore_path)
    print("the restore path is ", dvgg_restore_path)
    # 实例化一个全新的net,默认参数不用改
    net = TD3(a_dim,
              robot_state_dim,
              a_bound,
              replay_size=memory_size,

              # transition_num=5,
              # batch_size=batch_size,
              # memory_size=memory_size,
              )
    # 创建一个相机转换类
    camera_transform = CameraTransform(height=Origin_size, weight=Origin_size)
    # 序贯噪声类.可以修改的是max_sigma,值越大,噪声幅度越大,一般不能超过0.25,最小为0.05
    noise = Simple_noise(a_dim, -a_bound, a_bound,
                         dt=0.001, max_sigma=sigma)

    # 初始化几个变量
    successful_number = 0
    black_count = 0
    valid_loss_list = []
    episode_mean_loss = []
    valid_label_log = []
    valid_predict_log = []
    reward_logs = []

    ep_reward_list = []
    test_ep_reward_list = []

    drop_num = 0

    i = 0
    # 进入训练的主循环
    # for i in range(MAX_EPISODES):
    while i < MAX_EPISODES:
        episode_start = time.time()
        # obs = env.reset(reset_type=1, random_bias=0.1)
        # 这里本身最好用reset_type=1,random_bias为0.1,但是失效了.
        obs = env.reset(reset_type=1, random_bias=0.1)
        # noise.reset()

        ep_reward = 0
        real_ep_reward = 0

        # 获取state=perception+distance+shelter
        # distance,shelter = predict = dvgg.run(front, right)
        perception = np.hstack((obs['gripper_site_pos'], obs['joint_pos'], obs['robot-state']))

        front_to_vgg, right_to_vgg = img_trans(obs)
        now_predict = dvgg.run_ones(front_to_vgg,
                                    right_to_vgg,
                                    )
        s = np.hstack((perception, now_predict))

        a_loss_list, c_loss_list = [], []
        a_loss, c_loss = 0., 0.

        done_flag = False
        black_flag = False
        drop_flag = False
        # 进入内部循环
        for j in range(MAX_EP_STEPS):
            # 选择动作.
            if i < MAX_EPISODES:
                a = np.random.rand(a_dim) * action_high_bound
            else:
                a = net.get_action(s)
            # Add exploration noise
            # if noise_flag:
            # a = noise.add_noise(a)
            a = np.clip(a, -a_bound, a_bound)
            # 传入动作,获取观察值
            obs, r_total, done, info = env.step(a)
            # 这里的reward被我重新修改了,需要分解出来.修改的位置在print_task.py中
            # reward函数的return位置.如果维度不对,就在这里修改
            r, gripper_flag, step_done_flag = r_total
            # 其实是夹住的flag.
            if step_done_flag:
                done_flag = True

            # 保存net前视图的图片
            save_train_images2local(obs, i, j, logs_path, exp_names, save_end=Save_end)

            # 获取下一步真实标签

            next_perception = np.hstack((obs['gripper_site_pos'], obs['joint_pos'], obs['robot-state']))

            next_distances = get_distances(obs["joints_site_pos"],
                                           obs["gripper_site_pos"],
                                           obs["cube_pos"])

            next_shelter = get_shelters(obs["image"][0],
                                        obs["image"][3],
                                        obs["cube_pos"],
                                        obs["cube_quat"],
                                        env.cube_class,
                                        camera_transform)

            on_table_flag = check_on_table(obs["cube_pos"])
            if not on_table_flag:
                drop_flag = True
            # 判断是否黑图
            for p in obs["image"]:
                mean_pixel = np.mean(p)
                if mean_pixel < 2:
                    black_flag = True
                    break

            if black_flag:
                print("the episode is:", i)
                print("occ black image when the step is :", j)
                print("black_images_num:", black_count)
                if done_flag:
                    successful_number -= 1
                black_count += 1
                front_path = logs_path + exp_names + '/images/episode_front_' + str(i) + "_step_" + str(
                    j) + '_video.png'
                save_render_image(front_path, obs["image"][0])
                # import ipdb
                # ipdb.set_trace()
                print("*" * 30)

                break

            # 获取next_state的图片,并进行一样的预处理
            next_front_to_vgg, next_right_to_vgg = img_trans(obs)

            # 进行测试,获得每一步的测试结果!
            if Shelter_flag:
                next_total_labels = np.hstack((next_distances, next_shelter))
            else:
                next_total_labels = next_distances
            # 将预处理好的图片,传入预训练好的dvgg中,获得26个预测值
            next_predict = dvgg.run_ones(next_front_to_vgg,
                                         next_right_to_vgg,
                                         )
            next_total_predicts = next_predict
            # 将准确值和预测值保存到列表,最后存到本地.以便后期分析.
            valid_label_log.append(next_total_labels)
            valid_predict_log.append(next_total_predicts)
            # next_state的完整形态.包含perception,distance,shelter_info
            s_ = np.hstack((next_perception,
                            next_predict,
                            ))

            # 选择奖励函数!如果用遮挡信息的话,就用我们自己设计的奖励函数.
            # 还可以选择遮挡惩罚.
            if Shelter_flag:
                if i < 5000:
                    real_r = r
                    real_ep_reward += r
                    r = get_new_reward(next_predict[:-2], next_predict[-2:],
                                       step_done_flag, gripper_flag, punish_flag=punish_flag)
                    error_r = real_r - r

                    reward_logs.append([i, j, real_r, r, error_r])
                    if j % 100 == 0 or j == 199:
                        pred_dist_axis3 = next_predict[:-2][-3:]
                        # 设计一款新的遮挡惩罚!
                        # 当几乎被遮挡时,直接给予一个明确的惩罚,并且不考虑reach_reward.
                        pred_dist = np.linalg.norm(pred_dist_axis3)
                        real_dist_axis3 = obs["gripper_to_cube"]
                        real_dist = np.linalg.norm(real_dist_axis3)
                        real_r = 1 - np.tanh(10.0 * real_dist)
                else:
                    real_r = r
                    real_ep_reward += r
                    r = get_sparse_reward(next_predict[:-2], next_predict[-2:],
                                          step_done_flag, gripper_flag,
                                          punish_flag=punish_flag)
                    reward_logs.append([i, j, real_r, r, error_r])
            else:
                r = r

            # 保存转移元组
            # 只要完成夹取,就给予terminal=1!!!
            # 试试最终的结果会不会变好!
            if j == MAX_EP_STEPS - 1 or black_flag or drop_flag:
                net.store_transition((s, a, r, s_, 1))
            else:
                net.store_transition((s, a, r, s_, 0))

            # 置换state变量
            s = s_

            ep_reward += r

            if omit_drop_flag:
                if drop_flag:
                    # 清除掉之前的成功记录！
                    if done_flag:
                        successful_number -= 1
                    drop_num += 1
                    episode_time = time.time() - episode_start
                    print('Episode:', i, ' Reward: %0.5f' % ep_reward,
                          'Episode_time: %.3f' % episode_time,
                          'end step:', j,
                          "learn step:", net.learn_step, )
                    print("drop_num:", drop_num)
                    print("total_episode:", drop_num + i)
                    break

            # episode结束后的处理!
            if j == MAX_EP_STEPS - 1:

                ep_reward_list.append([ep_reward, real_ep_reward])
                for l in range(MAX_EP_STEPS):
                    # 如果是训练阶段,进入训练,根据batct_size选择间隔,如果是512,间隔8次训练一次
                    # 如果256,间隔四次.
                    # 学习率衰减,是手动设定的,可以设为固定学习率
                    # 加一个opencv的时间延迟.
                    learn_time_start = time.time()
                    net.learn(batch_size=batch_size)

                    learn_time = time.time() - learn_time_start
                    if learn_time > 0.1:
                        print("episode:", i)
                        print("step:", j)
                        print("learn_time:", learn_time)

                if drop_flag:
                    drop_num += 1
                    print("drop_num:", drop_num)
                    print("total_episode:", i)

                # 画出最后一个step的预测图!
                range_list = np.linspace(1, len(next_total_labels), len(next_total_labels))
                plt.plot(range_list, next_total_predicts, marker='o', label="total_predict")
                plt.plot(range_list, next_total_labels, marker='*', label="total_label")
                plt.title("the " + "episode " + str(i) + "step" + str(j))
                plt.legend(loc="upper left")
                plt.savefig(plt_save_path + '/epoch_' + str(i) + "step_" + str(j) + " total_label.jpg")
                plt.close()

                episode_time = time.time() - episode_start
                # 打印出夹住的次数
                if done_flag:
                    successful_number += 1
                    print('reward：', np.round(r, 3), "-" * 10, 'done-time：', successful_number)
                # 存入训练的数据,和测试的数据
                ep_r_list.append([ep_reward, net.learn_step, episode_time, successful_number, done_flag, j])

                episode_mean_loss.append(np.mean(valid_loss_list))

                # net的数据
                episode_time = time.time() - episode_start
                # 打印一些基本参数
                mean_step_r_list.append([ep_reward / (j + 0.00000001), j])
                plt_list.append([ep_reward, net.learn_step])
                end_step_r.append([ep_reward, j])

                print('Episode:', i,
                      ' Reward: %0.5f' % ep_reward,
                      ' real_Reward: %0.5f' % real_ep_reward,
                      'a_loss_mean：', np.round(np.mean(np.array(a_loss_list)), 4),
                      'c_loss_mean：', np.round(np.mean(np.array(c_loss_list)), 4),
                      'Episode_time: %.3f' % episode_time,
                      'end step:', j,
                      "learn step:", net.learn_step, )

                # 成功完成一个episode:

                # 增加测试部分!
                if i % 20 == 0:
                    test_ep_reward = test_agent(dvgg, net=net, env=env,
                                                n=Test_step)
                    test_ep_reward_list.append(test_ep_reward)
                    print("-" * 20)
                    print('Episode:', i, ' Reward: %i' % int(ep_reward),
                          'Test Reward: %4f' % int(test_ep_reward),
                          )
                    print("-" * 20)
                # 这里有一个迷惑,如果遇到意外:黑图/掉下去,提前结束,那么将会多保存一些数据!
                i += 1
                break
    # 保存dvgg基本验证信息
    valid_logs_path = logs_path + exp_names + '/dvgg_data/'
    valid_labels_array = np.array(valid_label_log)
    valid_predicts_array = np.array(valid_predict_log)
    valid_name = valid_logs_path + "net_data_predict"
    valid_labels_list = np.reshape(valid_labels_array, (-1, valid_labels_array.shape[-1]))
    valid_predicts_list = np.reshape(valid_predicts_array, (-1, valid_predicts_array.shape[-1]))
    print("valid_name:", valid_name)
    pd_data = pd.DataFrame(data=valid_labels_list)
    pd_data.to_csv(valid_name + '_labels.csv', encoding='gbk')
    pd_data = pd.DataFrame(data=valid_predicts_list)
    pd_data.to_csv(valid_name + '_predicts.csv', encoding='gbk')

    # episode_mean_accuracy_array = np.array(episode_mean_accuracy).reshape((len(episode_mean_accuracy), 1))
    episode_mean_loss_array = np.array(episode_mean_loss).reshape((len(episode_mean_loss), 1))

    plt.plot(episode_mean_loss_array, color="red", marker='o', linewidth=2.0, label="episode_mean_loss")
    plt.title("the " + "episode_mean_loss_array")
    plt.legend(loc="upper left")
    plt.savefig(plt_save_path + "/episode_mean_loss_array.jpg")
    plt.close("all")

    # 保存奖励函数记录
    if shelter_flag:
        name = ["episode", "step", "real_reward", "predict_reward", "reward_error"]
        pd_data = pd.DataFrame(columns=name, data=reward_logs)
        pd_data.to_csv(logs_path + exp_names + '/' + special_name + '_reward_logs.csv', encoding='gbk')

    # 存的训练部分的数据
    name = ['episode_reward', 'learning_step', 'episode_time', "successful_time", 'done', 'end-step']
    print("ep_r_list.shape", np.array(ep_r_list).shape)
    pd_data = pd.DataFrame(columns=name, data=ep_r_list)
    pd_data.to_csv(logs_path + exp_names + '/' + special_name + '_total_value.csv', encoding='gbk')
    print("ep_r_list.shape:", np.array(ep_r_list).shape)
    print("mean episode reward:", np.mean(np.array(ep_r_list)[:, 0]))

    # 存的测试部分的数据! test_plt_list[r_mean[0], train_episode, j]
    name = ['mean_reward']
    print("ep_r_list.shape", np.array(test_ep_reward_list).shape)
    pd_data = pd.DataFrame(columns=name, data=test_ep_reward_list)
    pd_data.to_csv(logs_path + exp_names + '/' + special_name + '_test_total_value.csv', encoding='gbk')

    plt_function(plt_list, logs_path + exp_names, special_name, y_label="Episode reward", show_flag=False)
    # 画出mean step reward
    plt_function(mean_step_r_list, logs_path + exp_names, special_name, 'mean step reward', show_flag=False)
    # 画出最后无噪声的几个图
    last_list = list(np.array(plt_list)[-No_noise_number - 200:, :2])
    plt_function(last_list, logs_path + exp_names, 'half no noise', y_label='Episode reward', show_flag=False)
    plt_end_step(end_step_r, logs_path + exp_names, show_flag=False)

    print('Running time: ', time.time() - t1)

    print("successful number:", successful_number)
    print("successful rate:", np.round(successful_number / float(MAX_EPISODES), 4))

    save_exp_name = ENV_NAME + '_' + special_name
    save_path = logs_path + save_exp_name + "/save_networks"
    print("save_path:", save_path)

    # net.save_simple_network(save_path)

    # plot 真假episode reward!
    ep_reward_list = np.array(ep_reward_list)
    range_list = np.linspace(1, len(ep_reward_list), len(ep_reward_list))
    plt.plot(range_list, ep_reward_list[:, 0], color="red", marker='*',
             linewidth=1.0, label="predict_reward")
    plt.plot(range_list, ep_reward_list[:, 1], color="blue", marker='o',
             linewidth=1.0, label="real_reward")

    plt.title("predict vs real reward")
    plt.legend(loc="upper left")
    plt.savefig(plt_save_path + "/predict_reward_vs_real_reward.jpg")
    plt.close("all")


if __name__ == "__main__":
    print("""            
            

    """)

    try:
        sigma = float(sys.argv[1])
    except:
        sigma = 0.18
    print("sigma:", sigma)
    print("sigma:", type(sigma))

    try:
        punish_flag = int(sys.argv[3])
        if punish_flag == 1:
            punish_flag = True
        else:
            punish_flag = False
    except:
        punish_flag = True
    print("punish_flag:", punish_flag)
    print("punish_flag:", type(punish_flag))

    if punish_flag:
        special_name = 'TD3_tune_dvgg_epoch' + net_name + '_episode_' + \
                       str(MAX_EPISODES) + '_net_per_punish-seed_' + str(seed)
    else:
        special_name = 'TD3_tune_dvgg_epoch' + net_name + '_episode_' + \
                       str(MAX_EPISODES) + '_net_per_no_punish-seed_' + str(seed)
    main(special_name=special_name,
         batch_size=512,
         memory_size=500000,
         sigma=sigma,
         )




