import matplotlib.pyplot as plt
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
env.seed(1)
np.random.seed(1)
tf.set_random_seed(1)
# EPISILON = 1  # Probability of doing a random move
GAMMA = 0.99  # Discounted future reward. How much we care about steps further in time
N_FEATURE = env.observation_space.shape[0]
N_ACTION = env.action_space.n


LR = 0.005



D = deque()  # initialize memory

with tf.variable_scope('eval-net'):
    tfx = tf.placeholder(tf.float32, [None, N_FEATURE], name='tfx')
    tfy = tf.placeholder(tf.int32, [None ], name='tfy')
    tfv = tf.placeholder(tf.float32, [None ])
    hid1 = tf.layers.dense(tfx, 10, activation=tf.nn.relu)
    out = tf.layers.dense(hid1, N_ACTION, activation=tf.nn.softmax)
    # neg_log_prob =
    # loss = tf.reduce_mean(-tf.reduce_sum(tf.log(out)*tfy, axis=1)*tfv)
    neg_log_prob = tf.reduce_sum(-tf.log(out) * tf.one_hot(tfy, N_ACTION), axis=1)
    loss = tf.reduce_mean(neg_log_prob * tfv)
    train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess= tf.Session()
sess.run(tf.global_variables_initializer())
saver= tf.train.Saver()


def choose_action(state):
    prob = sess.run(out,{tfx:state})
    action = np.random.choice(range(prob.shape[1]), p=prob.ravel())
    return action


def calculate_reward_decay(rewards):
    running_add = 0
    acu_rewards = np.zeros((rewards.shape[0]))

    for i in reversed(range(0,rewards.shape[0])):
        running_add = running_add*GAMMA + rewards[i]
        acu_rewards[i] = running_add
    # normalize
    # acu_rewards -= np.mean(acu_rewards)
    # acu_rewards /= np.std(acu_rewards)
    return acu_rewards


win_count = 0
rewards_history =[]
for epi in range(3500):
    steps = 0
    obs = env.reset()
    obs = obs[np.newaxis, :]

    while True:
        action = choose_action(obs)

        obs_new, reward, done, info =env.step(action)

        D.append([obs, action, reward, done])
        obs = obs_new[np.newaxis,:]
        steps += 1
        if done:
            memory = np.asarray(D)
            D.clear()
            inputs = np.vstack(memory[:,0])
            actions = np.hstack(memory[:,1])

            rewards = np.vstack(memory[:,2])
            acu_rewards = calculate_reward_decay(rewards)
            # rewards = np.tile(rewards,actions.shape[0])
            _ = sess.run(train_op,{tfx:inputs,tfy:actions,tfv:acu_rewards})
            rewards_history.append(steps)
            memory = []
            break
    if epi % 100 == 0:
        print('epi: {}-reward: {}| LR: {}'.format(epi, np.mean(rewards_history),LR))
        # if  np.mean(rewards_history)==200: break
        rewards_history = []
        LR /= 2
    # if epi % 100 ==0:
    #     print('epi-{}-win-{}'.format(epi,win_count))
saver.save(sess,'./CP_PG-lr {}'.format(epi) ,write_meta_graph=False)



