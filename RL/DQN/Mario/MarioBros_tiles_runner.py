
import matplotlib.pyplot as plt
from collections import deque            # For storing moves
import numpy as np
import random     # For sampling batches from the observations
import tensorflow as tf
import os
from subprocess import call

import gym
import gym_pull
# gym_pull.pull('github.com/ppaquette/gym-super-mario')
env = gym.make('ppaquette/SuperMarioBros-1-1-Tiles-v0')
# env.unwrapped
# #
# print(env.action_space)             #MultiDiscrete(6)--> list [ 1,1,1,0,1,1]
# print(env.observation_space)
# print(env.observation_space.high)   # Box(224,256,3 )
# print(env.observation_space.low)

# initail setting#
EPISILON = 0                           # Probability of doing a random move
gamma = 0.9                                # Discounted future reward. How much we care about steps further in time
mb_size = 64                              # Learning minibatch size
# n_feature = env.observation_space.shape[0]
n_action = env.action_space.shape
tot_steps = 1
D = deque() # initialize memory
MEMORY_SIZE = 10000
BATCH_SIZE = 50
REPLACE_ITER = 500
LR= 0.001
STATE_SIZE = 6
# LOGDIR =  os.path.join(os.getcwd(),'log_mario')

#
###function###

# # CNN-1
with tf.variable_scope('eval-net'):

    tfx = tf.placeholder(tf.float32,[None,6,13,16])
    flat = tf.reshape(tfx,[-1, 6*13*16])
    tfy = tf.placeholder(tf.float32,[None,6])
    hidden1 = tf.layers.dense(flat,128,activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1,64 , activation=tf.nn.relu)
    output = tf.layers.dense(hidden2,n_action,activation=None)
    loss = tf.reduce_mean(tf.squared_difference(output,tfy))

with tf.variable_scope('target-net'):

    tfx_t = tf.placeholder(tf.float32,[None,6,13,16])
    flat_t = tf.reshape(tfx_t,[-1, 6*13*16])
    tfy_t = tf.placeholder(tf.float32,[None,6])
    hidden1_t = tf.layers.dense(flat_t,128,activation=tf.nn.relu)
    hidden2_t = tf.layers.dense(hidden1_t,64 , activation=tf.nn.relu)
    output_t = tf.layers.dense(hidden2_t,n_action,activation=None)

train_op = tf.train.AdamOptimizer(LR).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess,tf.train.latest_checkpoint('./mario-checkpoint'))
# saver.restore(sess,'./mario-checkpoint/Mario-1720000')

# STARTING_EPOCH = 7
# fname = './mario-checkpoint/Mario-'+ str(STARTING_EPOCH)
# saver.restore(sess,fname)


tot_steps = 1
train_summary = []
cost_history = []
win_count = 0
max_reward = 0

def choose_action(EPISILON, state):
    if np.random.rand() <= EPISILON:  # random
        action = env.action_space.sample()
    else:  # learned action
        Q = sess.run(output, {tfx: state})
        Q = Q-np.mean(Q)
        action = np.round(sig(Q),0).reshape((6,)).astype(int).tolist()
        print(Q,action)
    return action

def sig(x):
    return 1/(1+np.exp(-x))

def get_state(state,observation_new):
    observation_new = observation_new[np.newaxis,:]
    # state as deque object
    state.append(observation_new)
    return np.vstack(state)[np.newaxis,:]



obs = env.reset()
obs = obs[np.newaxis,:] #(1,13,16)

# initialization
state_his = deque(maxlen=STATE_SIZE) #initial state
for _ in range(6):
    state_his.append(obs)
state = np.vstack(state_his)[np.newaxis,:] # (1,6,13,16)


print(sess.run(output,{tfx:state}))

# test= sess.run(output,{tfx:obs})


for epi in range(10):


    while True:

    # for step in range(200):
        action = choose_action(EPISILON, state)
        observation_new, reward, done, info = env.step(action)
        if done and info['distance'] <= 3500:
            reward = -10
        state_new = get_state(state_his, observation_new)
        D.append([state, action, reward, state_new])
        state = state_new

    # update
        tot_steps += 1

        # if done:

            # env.close()
            # call(['pkill', '-9', 'fceux'])
            # env = gym.make('ppaquette/SuperMarioBros-1-1-Tiles-v0')
            # env.reset()
            # break
            # env.change_level(new_level='ppaquette/SuperMarioBros-1-1-Tiles-v0')
            # break
print('end')



# write_log(LR,tot_steps)
