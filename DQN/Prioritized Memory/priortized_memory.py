import numpy as np
from collections import deque
import random
import gym
import tensorflow as tf


class Sumtree(object):
    data_pointer = 0

    def __init__(self, memory_size, batch_size,
                 alpha,
                 beta):
        # initialize tree structure

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.tree = np.zeros(2 * memory_size - 1)
        self.tree[memory_size - 1] = 1  # initialize p
        self.alpha = alpha
        self.beta = beta
        # self.data = np.zeros(memory_size,dtype = object)

        # [--------------Parent nodes-------------][-------leaves to record priority-------]
        #            size: memory_size - 1                       size: memory_size
        # ex.memory size = 100
        # [0                                   98][99                                     198]

    def add(self, p):
        tree_idx = self.data_pointer + self.memory_size - 1
        # self.tree[tree_idx] = p
        self.data_pointer += 1
        self.update(tree_idx, p)
        self.data_pointer = self.data_pointer % self.memory_size

    def sample(self):
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
        # find a random value ,v, in each interval, defined by p/sum(p)
        rand_interval = self.total_p / self.batch_size
        data_idx = np.zeros(self.batch_size, dtype=int)
        ISWeight = np.empty(self.batch_size, dtype=float)
        for i in range(self.batch_size):
            v = np.random.uniform(rand_interval * i, rand_interval * (i + 1))
            parent_idx = 0
            pick_idx = 0
            while True:
                # [--------------Parent nodes-------------][-------leaves to record priority-------]
                #             size: memory_size - 1                       size: memory_size
                left_tree_idx = 2 * parent_idx + 1
                right_tree_idx = left_tree_idx + 1

                if left_tree_idx >= len(self.tree):
                    pick_idx = parent_idx
                    break
                else:
                    if v <= self.tree[left_tree_idx]:
                        parent_idx = left_tree_idx
                    else:
                        parent_idx = right_tree_idx
                        v -= self.tree[left_tree_idx]
            data_idx[i] = pick_idx
            prob_j = self.tree[data_idx[i]] / self.total_p
            ISWeight[i] = np.power((prob_j / np.max(self.tree[self.memory_size - 1:])), self.beta * -1)

        return data_idx, ISWeight

    def update(self, tree_idx, p):
        # update the whole tree frame once p is changed

        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    @property
    def total_p(self):
        return self.tree[0]  # np.sum(self.tree[self.memory_size-1:self.memory_size*2])


class Memory(object):
    def __init__(self, memory_size, batch_size,
                 alpha=0.6,
                 beta=0.4,
                 epsilon=0.01):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.data = np.zeros(self.memory_size, dtype=object)
        self.sumtree = Sumtree(self.memory_size,
                               self.batch_size, self.alpha, self.beta)

    def sample(self):
        data_idx, ISWeight = self.sumtree.sample()
        sample_idx = data_idx - self.memory_size - 1
        sample_data = np.vstack(self.data[sample_idx])
        return sample_data, data_idx, ISWeight

    def store(self, transition):
        transition = np.asarray(transition)
        self.data[self.sumtree.data_pointer] = transition
        # self.data.append(transition)
        p = np.max(self.sumtree.tree[-self.sumtree.memory_size:])
        self.sumtree.add(p)

    def batch_update(self, tree_idx, td_error):  ##clip td error?
        abs_td_error = np.abs(td_error)
        abs_td_error += self.epsilon
        p = np.power(abs_td_error, self.alpha)
        for ti, pi in zip(tree_idx, p):
            self.sumtree.update(ti, pi)


class DQN_prioritize(object):
    def __init__(self, n_f, n_a, lr, greedy, memory_size, batch_size,
                 alpha=0.6,
                 beta=0.4,
                 epsilon=0.01):
        self.n_f = n_f
        self.n_a = n_a
        self.lr = lr
        self.greedy = greedy
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = Memory(self.memory_size, batch_size, alpha, beta, epsilon)

        # network
        self.tfs = tf.placeholder(tf.float32, [None, self.n_f])  # input state
        self.tfs_ = tf.placeholder(tf.float32, [None, self.n_f])  # input observation
        self.tfy = tf.placeholder(tf.float32, [None, self.n_a])  # input Q for each action
        self.tfw = tf.placeholder(tf.float32, [None, 1])  # input for isWeight
        self.td_error = tf.placeholder(tf.float32, [None, 1])  # input for td_error

        with tf.variable_scope('eval'):
            self.q = self._build_net(self.tfs)
        with tf.variable_scope('target'):
            self.q_ = self._build_net(self.tfs_)

        self.loss = tf.reduce_mean(self.tfw * tf.squared_difference(self.q, self.tfy)) # * self.td_error
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.update_target()

    def _build_net(self, s):
        hid1 = tf.layers.dense(s, 20, activation=tf.nn.relu)
        return tf.layers.dense(hid1, self.n_a, activation=None)

    def update_target(self):
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval')
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

    def choose_action(self, state):
        if np.random.rand() >= self.greedy:
            action = env.action_space.sample()
        else:
            Q = self.sess.run(self.q, {self.tfs: state})
            action = np.argmax(Q)
        return action

    def learn(self):
        sample_data, self.data_idx, ISWeight = self.memory.sample()
        inputs = np.vstack(sample_data[:, 0])
        actions = sample_data[:, 1].astype(int)
        rewards = sample_data[:, 2]
        dones = sample_data[:, 3]
        next_inputs = np.vstack(sample_data[:, 4])

        q = self.sess.run(self.q, {self.tfs: inputs})
        q_, qeval_next = self.sess.run([self.q_, self.q], {self.tfs_: next_inputs, self.tfs: next_inputs})
        q_eval_next_index = np.argmax(qeval_next, axis=1)

        i = np.arange(self.batch_size, dtype=int)

        td_error = rewards + (1 - dones) * GAMMA * q_[i, q_eval_next_index] - q[i, actions]
        td_error_clipped = np.clip(td_error, -1, 1)
        # Bellman's eq
        q[i, actions] = rewards + (1 - dones) * GAMMA * q_[i, q_eval_next_index]

        _, loss_ = self.sess.run([self.train_op, self.loss],
                                 {self.tfs: inputs, self.tfy: q, self.tfw: ISWeight[:, np.newaxis],
                                  self.td_error: td_error_clipped[:, np.newaxis]})
        self.loss_ = np.mean(loss_)
        self.memory.batch_update(self.data_idx, td_error_clipped)

    def store_memory(self, transition):
        self.memory.store(transition)


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    #

    ###hyperparameter###
    n_a = env.action_space.n
    n_f = env.observation_space.shape[0]
    ALPHA = 0.4
    BETA = 0.6
    GREEDY = 0.5
    LR = 0.01
    GAMMA = 0.99
    MEMORY_SIZE = 4
    BATCH_SIZE = 2

    s = env.reset()
    s = s[np.newaxis, :]
    s1 = s
    tot_steps = 1
    rwd_his = []
    cost_his=[]

    Agent = DQN_prioritize(n_f, n_a, LR, GREEDY, MEMORY_SIZE, BATCH_SIZE)

    for epi in range(2000):
        running_r = 0
        while True:

            a = Agent.choose_action(s)
            s_, r, d, info = env.step(a)
            s_ = s_[np.newaxis, :]
            Agent.store_memory([s, a, r, d * 1, s_])  # convert done: bool->int
            tot_steps += 1
            s = s_

            running_r += r


            if tot_steps >= MEMORY_SIZE:
                Agent.learn()
                cost_his.append(Agent.loss_)
                # print(tot_steps)
                # print(Agent.data_idx)
                # print(Agent.memory.sumtree.tree)
                #print(Agent.memory.data[:][0])
                # print('loss:{}'.format(Agent.loss_))
            if d:
                s = env.reset()
                s = s[np.newaxis, :]
                rwd_his.append(running_r)
                break

        if epi % 1 == 0:
            print('epi:{}|rwd:{:04.2f}|loss:{}'.format(epi,np.mean(rwd_his),np.mean(Agent.loss_)))
            print('Q_int:{}'.format(Agent.sess.run(Agent.q,{Agent.tfs:s1})))
            rwd_his=[]
            cost_his=[]
    print("done")
