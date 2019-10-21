"""
从最原始的成功的DDPG_OU_noise_memory_list.py抽取出来.
删掉噪声部分.
剔除memory部分.
便于以后加per.
"""

import tensorflow as tf
import numpy as np
import sys
from memory.simple_memory import Memory


#####################  hyper parameters  ####################
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement


###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, transition_num=4, restore_flag=False, batch_size=32, memory_size=100000):
        self.transition_num = transition_num
        self.memory = Memory(memory_size=memory_size,
                             batch_size=batch_size,
                             transition_num=transition_num,
                             )
        self.batch_size = batch_size

        self.pointer = 0
        self.learn_step = 0
        self.restore_flag = restore_flag

        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.actor_lr = tf.placeholder(tf.float32, shape=[], name='actor_lr')
        self.critic_lr = tf.placeholder(tf.float32, shape=[], name='critic_lr')

        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # hard_replace
        self.hard_replace = [tf.assign(t, e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.c_loss = td_error
        self.ctrain = tf.train.AdamOptimizer(self.critic_lr).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.a_loss = a_loss
        self.atrain = tf.train.AdamOptimizer(self.actor_lr).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def store_transition(self, transition):
        self.memory.store(transition)

    def learn(self, actor_lr_input, critic_lr_input, output_loss_flag=False):
        # soft target replacement
        self.sess.run(self.soft_replace)
        # 加上terminal信息
        if self.transition_num==5:
            bs, ba, br, bs_, bt = self.memory.sample()
        if self.transition_num==4:
            bs, ba, br, bs_ = self.memory.sample()

        self.learn_step += 1

        if output_loss_flag:
            _, a_loss = self.sess.run([self.atrain, self.a_loss], {self.S: bs, self.actor_lr: actor_lr_input})
            _, c_loss = self.sess.run([self.ctrain, self.c_loss],
                            {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.critic_lr: critic_lr_input})
            return a_loss, c_loss
        else:
            self.sess.run(self.atrain, {self.S: bs, self.actor_lr: actor_lr_input})
            self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.critic_lr: critic_lr_input})



    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 300, activation=tf.nn.relu, name='l1', trainable=trainable)
            new_actor_layer = tf.layers.dense(net, 200, activation=tf.nn.relu, name='new_actor_layer', trainable=trainable)
            a = tf.layers.dense(new_actor_layer, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 400
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            new_critic_layer = tf.layers.dense(net, 300, activation=tf.nn.relu, name='new_critic_layer',
                                              trainable=trainable)
            return tf.layers.dense(new_critic_layer, 1, trainable=trainable)  # Q(s,a)

    def load_step_network(self, saver, load_path):
        checkpoint = tf.train.get_checkpoint_state(load_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(self.sess, tf.train.latest_checkpoint(load_path))
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
            self.learn_step = int(checkpoint.model_checkpoint_path.split('-')[-1])

        else:
            print("Could not find old network weights")

    def save_step_network(self, time_step, saver, save_path):
        saver.save(self.sess, save_path + 'network', global_step=time_step,
                   write_meta_graph=False)

    def load_simple_network(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(path))
        print("restore model successful")

    def save_simple_network(self, save_path):
        saver = tf.train.Saver()
        saver.save(self.sess, save_path=save_path + "/params", write_meta_graph=False)
