import numpy as np
import tensorflow as tf
import gym
import time
import sys
sys.path.append("../")
from td3_sp import core
from td3_sp.core import get_vars, mlp_actor_critic


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


class TD3:
    def __init__(self,
                 a_dim, obs_dim, a_bound,
                 mlp_actor_critic=core.mlp_actor_critic,
                 ac_kwargs=dict(), seed=0,

                 replay_size=int(1e6), gamma=0.99,
                 polyak=0.995, pi_lr=1e-3, q_lr=1e-3,
                 batch_size=100,
                 # start_steps=10000,
                 act_noise=0.1, target_noise=0.2,
                 noise_clip=0.5, policy_delay=2,
                 # max_ep_len=1000,
                 # logger_kwargs=dict(), save_freq=1
                 ):

        self.learn_step = 0

        self.obs_dim = obs_dim
        self.act_dim = a_dim
        self.act_limit = a_bound
        self.policy_delay = policy_delay
        self.action_noise = act_noise

        # Share information about action space with policy architecture
        ac_kwargs['action_space'] = a_bound

        # Inputs to computation graph
        self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = core.placeholders(obs_dim, a_dim, obs_dim, None, None)

        # Main outputs from computation graph
        with tf.variable_scope('main'):
            self.pi, self.q1, self.q2, self.q1_pi = mlp_actor_critic(self.x_ph, self.a_ph, **ac_kwargs)

        # Target policy network
        with tf.variable_scope('target'):
            pi_targ, _, _, _ = mlp_actor_critic(self.x2_ph, self.a_ph, **ac_kwargs)

        # Target Q networks
        with tf.variable_scope('target', reuse=True):

            # Target policy smoothing, by adding clipped noise to target actions
            epsilon = tf.random_normal(tf.shape(pi_targ), stddev=target_noise)
            epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = tf.clip_by_value(a2, -self.act_limit, self.act_limit)

            # Target Q-values, using action from target policy
            _, q1_targ, q2_targ, _ = mlp_actor_critic(self.x2_ph, a2, **ac_kwargs)

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=self.act_dim, size=replay_size)

        # Count variables
        var_counts = tuple(core.count_vars(scope)
                           for scope in ['main/pi',
                                         'main/q1',
                                         'main/q2',
                                         'main'])
        print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n' % var_counts)

        # Bellman backup for Q functions, using Clipped Double-Q targets
        min_q_targ = tf.minimum(q1_targ, q2_targ)
        backup = tf.stop_gradient(self.r_ph + gamma * (1 - self.d_ph) * min_q_targ)

        # TD3 losses
        self.pi_loss = -tf.reduce_mean(self.q1_pi)
        q1_loss = tf.reduce_mean((self.q1 - backup) ** 2)
        q2_loss = tf.reduce_mean((self.q2 - backup) ** 2)
        # 为啥这里的loss是加起来的?
        self.q_loss = q1_loss + q2_loss

        # Separate train ops for pi, q
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
        q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
        self.train_pi_op = pi_optimizer.minimize(self.pi_loss,
                                                 var_list=get_vars('main/pi'))
        # 这里的参数,怎么是总的q?
        # 难道这里的字符串只需要匹配就好了?
        self.train_q_op = q_optimizer.minimize(self.q_loss,
                                               var_list=get_vars('main/q'))

        # Polyak averaging for target variables
        self.target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                                       for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        # Initializing targets to match main variables
        target_init = tf.group([tf.assign(v_targ, v_main)
                                for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(target_init)

    def get_action(self, s, noise_scale=0):
        if not noise_scale:
            noise_scale = self.action_noise
        a = self.sess.run(self.pi, feed_dict={self.x_ph: s.reshape(1, -1)})[0]
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    def store_transition(self, transition):
        (s, a, r, s_, done) = transition
        self.replay_buffer.store(s, a, r, s_, done)

    def test_agent(self, env, max_ep_len=1000, n=5):
        ep_reward_list = []
        for j in range(n):
            s = env.reset()
            ep_reward = 0
            for i in range(max_ep_len):
                # Take deterministic actions at test time (noise_scale=0)
                s, r, d, _ = env.step(self.choose_action(s))
                ep_reward += r
            ep_reward_list.append(ep_reward)
        mean_ep_reward = np.mean(np.array(ep_reward_list))
        return mean_ep_reward

    def learn(self, batch_size=100):

        batch = self.replay_buffer.sample_batch(batch_size)
        feed_dict = {self.x_ph: batch['obs1'],
                     self.x2_ph: batch['obs2'],
                     self.a_ph: batch['acts'],
                     self.r_ph: batch['rews'],
                     self.d_ph: batch['done']
                     }
        q_step_ops = [self.q_loss, self.q1, self.q2, self.train_q_op]
        outs = self.sess.run(q_step_ops, feed_dict)

        if self.learn_step % self.policy_delay == 0:
            # Delayed policy update
            outs = self.sess.run([self.pi_loss, self.train_pi_op, self.target_update],
                                 feed_dict)
        self.learn_step += 1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='td3')
    args = parser.parse_args()

    td3(lambda: gym.make(args.env), mlp_actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        )

