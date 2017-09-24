
import matplotlib.pyplot as plt
from collections import deque            # For storing moves
import numpy as np
import gym                                # To train our network
import random     # For sampling batches from the observations
import tensorflow as tf
import os

# env = gym.make('SuperMarioBros-1-1-v0')
env = gym.make('MountainCar-v0')          # Choose game (any in the gym should work)
env = env.unwrapped

# print parameter
# print(env.action_space)
# print(env.observation_space)  #Box(2,)
# print(env.observation_space.high) # [0.6 0.07]
# print(env.observation_space.low)

# initail setting#
EPISILON = 0.00                          # Probability of doing a random move
gamma = 0.9                                # Discounted future reward. How much we care about steps further in time
mb_size = 32                              # Learning minibatch size
n_feature = env.observation_space.shape[0]
n_action = env.action_space.n
tot_steps = 1
MEMORY_SIZE = 10000
# LOGDIR =  os.path.join(os.getcwd(),'log_mc')
 # D = deque() # initialize memory
D2 = deque(maxlen=MEMORY_SIZE) # initialize memory



with tf.variable_scope('eval-net'):
    tfx = tf.placeholder(tf.float32,[None,2],name='tfx')
    tfy = tf.placeholder(tf.float32,[None,3],name='tfy')
    hid1 = tf.layers.dense(tfx,20,activation=tf.nn.relu)
    out = tf.layers.dense(hid1,3,activation=None)
    # loss = tf.losses.mean_squared_error(labels=tfy,predictions=out)


# target
with tf.variable_scope('target-net'):
    tfx_t = tf.placeholder(tf.float32,[None,2])
    tfy_t = tf.placeholder(tf.float32,[None,3])
    hid1_t = tf.layers.dense(tfx_t,20,activation=tf.nn.relu)
    out_t = tf.layers.dense(hid1_t,3,activation=None)

sess = tf.Session()
saver = tf.train.Saver()
# sess.run(tf.global_variables_initializer())

saver.restore(sess,'./MC2input-129873')


train_summary = []
cost_history =[]
win_count =0


###functions###

def choose_action(epsilon,state):
    if np.random.rand() <= epsilon:  # random
        action = np.random.randint(0, n_action)
    else:  # learned action
        Q = sess.run(out, {tfx: state})
        action = np.argmax(Q)
    return action

def get_state(obs,state):
    new_state = np.zeros((8))
    new_state[:2] = obs
    new_state[2:] = state[0][:6]
    return np.expand_dims(new_state, axis = 0)

inp = np.zeros((1,2))
# o = sess.run(out,{tfx:inp})
# print(o)

for epi in range(10):

    obs = env.reset()
    obs = np.expand_dims(obs,axis=0)
    # state = np.expand_dims(np.hstack((obs,obs,obs,obs)),axis=0)
    # initialize memory buffer
    # memory = np.zeros((MEMORY_SIZE, state.shape[1] * 2 + 2 + 1))  # state*2 + act+ rwd + done

    while True:
    # for i in range(30000):
    #     env.render()

        action=choose_action(EPISILON,obs)

        observation_new, reward, done, info = env.step(action)
        observation_new = np.expand_dims(observation_new,axis =0)
        obs = observation_new
        tot_steps += 1
        if tot_steps % 10000 ==0:
            print(tot_steps)
        if done:
            print(tot_steps)
            break


###################
# env = gym.make('MountainCar-v0')          # Choose game (any in the gym should work)
# env = env.unwrapped
#
# # print parameter
# # print(env.action_space)
# # print(env.observation_space)  #Box(2,)
# # print(env.observation_space.high) # [0.6 0.07]
# # print(env.observation_space.low)
#
# # initail setting#
# EPISILON = 0.00                           # Probability of doing a random move
# gamma = 0.9                                # Discounted future reward. How much we care about steps further in time
# mb_size = 64                              # Learning minibatch size
# n_feature = env.observation_space.shape[0]
# n_action = env.action_space.n
# tot_steps = 1
# MEMORY_SIZE = 10000
# # LOGDIR =  os.path.join(os.getcwd(),'log_mc')
#  # D = deque() # initialize memory
# D2 = deque(maxlen=MEMORY_SIZE) # initialize memory
# env.seed(21)
#
#
# with tf.variable_scope('eval-net'):
#     tfx = tf.placeholder(tf.float32,[None,8],name='tfx')
#     tfy = tf.placeholder(tf.float32,[None,3],name='tfy')
#     hid1 = tf.layers.dense(tfx,20,activation=tf.nn.relu)
#     out = tf.layers.dense(hid1,3,activation=None)
#     loss = tf.reduce_mean(tf.squared_difference(out, tfy))
#     train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
#
#
# # target
# with tf.variable_scope('target-net'):
#     tfx_t = tf.placeholder(tf.float32,[None,8])
#     tfy_t = tf.placeholder(tf.float32,[None,3])
#     hid1_t = tf.layers.dense(tfx_t,20,activation=tf.nn.relu)
#     out_t = tf.layers.dense(hid1_t,3,activation=None)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# saver = tf.train.Saver()
# saver.restore(sess,'./MC-66343')
#
#
#
# def choose_action(epsilon,state):
#     if np.random.rand() <= epsilon:  # random
#         action = np.random.randint(0, n_action)
#     else:  # learned action
#         Q = sess.run(out, {tfx: state})
#         action = np.argmax(Q)
#     return action
#
# def get_state(obs,state):
#     new_state = np.zeros((8))
#     new_state[:2] = obs
#     new_state[2:] = state[0][:6]
#     return np.expand_dims(new_state, axis = 0)
#
#
# for epi in range(20):
#
#     obs = env.reset()
#     # obs = np.expand_dims(obs,axis=0)
#     state = np.expand_dims(np.hstack((obs,obs,obs,obs)),axis=0)
#     # initialize memory buffer
#
#     while True:
#     # for i in range(500):
#     #     env.render()
#
#         action=choose_action(EPISILON,state)
#
#         observation_new, reward, done, info = env.step(action)
#
#         state_new = get_state(observation_new,state) # state_new.shape= (1,8)
#         state = state_new
#         tot_steps += 1
#         if tot_steps % 10000 ==0:
#             print(tot_steps)
#         if done:
#             print(tot_steps)
#             break
