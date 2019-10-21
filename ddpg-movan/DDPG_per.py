"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np


#####################  hyper parameters  ####################


LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement


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


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = list(np.zeros(capacity, dtype=object))  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, transition):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = transition  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.full_flag = False

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        # n就是batch size！
        # np.empty()这是一个随机初始化的一个矩阵！
        b_idx, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1))
        b_memory = []
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        if min_prob == 0:
            min_prob = 0.00001
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i] = idx
            b_memory.append(data)

        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

###############################DDPG####################################


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, exp_path,
                 restore_flag=False,
                 batch_size=512,
                 per_batch_size=32,
                 memory_size=100000,
                 per_memory_size=20000):
        self.memory_size = memory_size
        self.memory = []
        self.per_memory = Memory(capacity=per_memory_size)
        self.per_memory_size = self.per_memory.tree.capacity
        self.pointer = 0
        self.per_pointer = 0

        self.batch_size = batch_size
        self.per_batch_size = per_batch_size
        self.exp_path = exp_path
        print("self.exp_path", self.exp_path)

        self.learn_step = 0
        self.restore_flag = restore_flag

        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.actor_lr = tf.placeholder(tf.float32, shape=[], name='actor_lr')
        self.critic_lr = tf.placeholder(tf.float32, shape=[], name='critic_lr')

        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

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
        # td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.abs_errors = tf.reduce_sum(tf.abs(q_target - q), axis=1)  # for updating Sumtree
        self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(q_target, q))
        self.ctrain = tf.train.AdamOptimizer(self.critic_lr).minimize(self.loss, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.actor_lr).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self, actor_lr_input, critic_lr_input, per_flag=True):
        # soft target replacement
        self.sess.run(self.soft_replace)

        if per_flag:
            tree_idx, batch_memory, ISWeights = self.per_memory.sample(self.per_batch_size)
            batch_states, batch_actions, batch_rewards, batch_states_ = [], [], [], []
            for i in range(self.per_batch_size):
                batch_states.append(batch_memory[i][0])
                batch_actions.append(batch_memory[i][1])
                batch_rewards.append(batch_memory[i][2])
                batch_states_.append(batch_memory[i][3])

            bs = np.array(batch_states)
            ba = np.array(batch_actions)
            batch_rewards = np.array(batch_rewards)
            bs_ = np.array(batch_states_)
            br = batch_rewards[:, np.newaxis]
        else:
            bs, ba, br, bs_ = self.sample_memory()

        # print("br:", br)

        self.sess.run(self.atrain, {self.S: bs, self.actor_lr: actor_lr_input})
        _, abs_errors, cost = self.sess.run([self.ctrain, self.abs_errors, self.loss],
                      {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.critic_lr: critic_lr_input,
                       self.ISWeights: ISWeights})

        self.per_memory.batch_update(tree_idx, abs_errors)  # update priority
        # print("lr:", self.sess.run(self.actor_lr, {self.actor_lr: actor_lr_input}))

        self.learn_step += 1

    def store_transition(self, s, a, r, s_):
        self.per_memory.store(transition=[s, a, r, s_])
        self.per_pointer = self.per_memory.tree.data_pointer
        if len(self.memory) >= self.memory_size:
            del self.memory[0]
        self.memory.append([s, a, r, s_])
        self.pointer = len(self.memory)

    def sample_memory(self):
        if len(self.memory) < self.memory_size:
            indices = np.random.choice(len(self.memory), size=self.batch_size)
        else:
            indices = np.random.choice(self.memory_size, self.batch_size)
        batch_states, batch_actions, batch_rewards, batch_states_ = [], [], [], []
        for i in indices:
            batch_states.append(self.memory[i][0])
            batch_actions.append(self.memory[i][1])
            batch_rewards.append(self.memory[i][2])
            batch_states_.append(self.memory[i][3])

        batch_states = np.array(batch_states)
        batch_actions = np.array(batch_actions)
        batch_rewards = np.array(batch_rewards)
        batch_states_ = np.array(batch_states_)
        batch_rewards = batch_rewards[:, np.newaxis]
        return batch_states, batch_actions, batch_rewards, batch_states_

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

    def load_network(self, saver, load_path):
        checkpoint = tf.train.get_checkpoint_state(load_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            # self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            saver.restore(self.sess, tf.train.latest_checkpoint(load_path))
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
            self.learn_step = int(checkpoint.model_checkpoint_path.split('-')[-1])

        else:
            print("Could not find old network weights")

    def save_network(self, time_step, saver, save_path):
        saver.save(self.sess, save_path + 'network', global_step=time_step,
                   write_meta_graph=False)




###############################  training  ####################################

