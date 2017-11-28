# import matplotlib.pyplot as plt
from collections import deque  # For storing moves
import numpy as np
import gym  # To train our network
import random  # For sampling batches from the observations
import tensorflow as tf
import os
from pathlib import Path
import pickle

env = gym.make('CartPole-v0')  # Choose game (any in the gym should work)
# env = env.unwrapped


EPISILON = 1  # Probability of doing a random move
GAMMA = 0.99  # Discounted future reward. How much we care about steps further in time
MB_SIZE = 64  # Learning minibatch size
N_FEATURE = env.observation_space.shape[0]
N_ACTION = env.action_space.n
BIAS = 100
tot_steps = 1

LR = 0.0001


def choose_action(epsilon, state):
    prob = sess.run(out,{tfx:state})
    action = np.random.choice(range(prob.shape[1]), p=prob.ravel())
    return action

with tf.variable_scope('eval-net'):
    tfx = tf.placeholder(tf.float32, [None, N_FEATURE], name='tfx')
    tfy = tf.placeholder(tf.float32, [None, N_ACTION], name='tfy')
    tfv = tf.placeholder(tf.float32, [None, 1])
    hid1 = tf.layers.dense(tfx,10, activation=tf.nn.relu)
    out = tf.layers.dense(hid1, N_ACTION, activation=tf.nn.softmax)
    neg_log_prob = tf.reduce_sum(-tf.log(out) * tfy, axis=1)
    loss = tf.reduce_mean(neg_log_prob * tfv)
    train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
# sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, './CP_PG-lr 3499')


win_count = 0

for epi in range(10):
    steps = 0
    obs = env.reset()
    obs = obs[np.newaxis, :]
    running_rewards = 0
    while True:
        env.render()
        action = choose_action(EPISILON, obs)

        obs_new, reward, done, info = env.step(action)
        running_rewards += reward
        obs = obs_new[np.newaxis,:]
        steps += 1
        if done:
            print('End | Rewards: {}'.format(running_rewards))
            break
