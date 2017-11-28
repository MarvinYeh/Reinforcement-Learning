'''
modiefied from tutorials from: https://github.com/MorvanZhou
'''


import numpy as np
import tensorflow as tf
import gym
from collections import deque



#########hyperparameter########
MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32


RENDER = False
ENV_NAME = 'Pendulum-v0'


#########hyperparameter########

#consturct graph for actor and critic


class DDPG(object):
    def __init__(self, n_f, n_a, a_bound):
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.sess = tf.Session()
        self.n_f, self.n_a, self.a_bound = n_f, n_a, a_bound
        self.s = tf.placeholder(tf.float32, [None, n_f])
        self.s_ = tf.placeholder(tf.float32, [None, n_f])
        self.r = tf.placeholder(tf.float32, [None, 1])


        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.s, scope='eval', trainable=True)
            a_ = self._build_a(self.s_, scope='target', trainable=False)

        with tf.variable_scope('Critic'):
            self.q = self._build_c(self.s, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.s_, self.a_, scope='target', trainable=False)

        # network parameter
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]
        q_target = self.r + GAMMA *q_
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.q)
        self.ctrain = tf.train.AdamOptimizer(LR_A).minimize(td_error, var_list=self.ce_params)

        a_loss = -tf.reduce_mean(q)
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss,var_list=self.ae_params)
        self.sess.run(tf.global_variables_initializer())




    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            hid1 = tf.layers.dense(s, 30, activation=tf.nn.relu, trainable=trainable)
            a = tf.layers.dense(hid1, self.n_a, tf.nn.tanh, trainable=trainable)
            return tf.multiply(a,self.a_bound)

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            hid1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(hid1, 1, trainable)





if __name__ == "__main__":
    env = gym.make(ENV_NAME)

    n_f = env.observation_space.shape[0]
    n_a = env.action_space.shape[0]
    a_bound = env.action_space.high

#