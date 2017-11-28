
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
tot_steps = 1
D = deque() # initialize memory
MEMORY_SIZE = 10000
LR= 0.001
STATE_SIZE = 2
# LOGDIR =  os.path.join(os.getcwd(),'log_mario')


act_option = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0],
              [1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 1, 0],
              [1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 1],
              [1, 0, 0, 0, 1, 1], [0, 1, 0, 0, 1, 1], [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 1, 1],
              [1, 1, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0], [0, 1, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0],
              [1, 1, 0, 0, 1, 0], [1, 0, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0], [0, 0, 1, 1, 1, 0],
              [1, 1, 0, 0, 0, 1], [1, 0, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1], [0, 0, 1, 1, 0, 1],
              [1, 1, 0, 0, 1, 1], [1, 0, 0, 1, 1, 1], [0, 1, 1, 0, 1, 1], [0, 0, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 1]]


n_a = len(act_option)
# act_option =np.asarray(act_option)

# # CNN-1
with tf.variable_scope('eval-net'):
    tfx = tf.placeholder(tf.float32, [None, STATE_SIZE, 13, 16])
    flat = tf.reshape(tfx, [-1, STATE_SIZE * 13 * 16])
    tfy = tf.placeholder(tf.float32, [None,n_a])
    hidden1 = tf.layers.dense(flat, 128, activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, 64, activation=tf.nn.relu)
    output = tf.layers.dense(hidden2, n_a, activation=None)
    loss = tf.reduce_mean(tf.squared_difference(output, tfy))

with tf.variable_scope('target-net'):
    tfx_t = tf.placeholder(tf.float32, [None, STATE_SIZE, 13, 16])
    flat_t = tf.reshape(tfx_t, [-1, STATE_SIZE * 13 * 16])
    tfy_t = tf.placeholder(tf.float32, [None,n_a])
    hidden1_t = tf.layers.dense(flat_t, 128, activation=tf.nn.relu)
    hidden2_t = tf.layers.dense(hidden1_t, 64, activation=tf.nn.relu)
    output_t = tf.layers.dense(hidden2_t, n_a, activation=None)

train_op = tf.train.AdamOptimizer(LR).minimize(loss)
sess = tf.Session()
saver = tf.train.Saver(max_to_keep=200)
sess.run(tf.global_variables_initializer())
saver.restore(sess,'./mario-checkpoint/Mario-3000')

STARTING_STEP = 0
# fname = './mario-logs/Mario-'+ str(STARTING_STEP)
# saver.restore(sess,fname)

# assign the same parameter
e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval-net')
t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target-net')
sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

##actionlist##


#
# # initialize memory
tot_steps = 1
D = deque(maxlen=10000)
#
all_running = []
cost_history = []
max_reward = 0


def choose_action(EPISILON, state):
    if np.random.rand() <= EPISILON:  # random
        action_idx = np.random.randint(n_a)
        action = act_option[action_idx]

    else:  # learned action
        Q = sess.run(output, {tfx: state})
        action_idx = np.argmax(Q)
        action = act_option[action_idx]
    return action, action_idx


def write_log(LR, tot_steps):
    with open('./mario-log/training_log.txt', 'a') as file:
        file.write('LR:{}- Steps:{}\n'.format(LR, tot_steps))


def get_state(state, observation_new):
    observation_new = observation_new[np.newaxis, :]
    # state as deque object
    state.append(observation_new)
    return np.vstack(state)[np.newaxis, :]



obs = env.reset()
obs = obs[np.newaxis,:] #(1,13,16)

# initialization
state_his = deque(maxlen=STATE_SIZE)  # initial state
for _ in range(STATE_SIZE):
    state_his.append(obs)
state = np.vstack(state_his)[np.newaxis, :]  # (1,6,13,16)
s1 = state #for checking q value


for epi in range(5):
    running_reward = 0

    while True:

        action, action_idx = choose_action(EPISILON, state)
        observation_new, reward, done, info = env.step(action)

        state_new = get_state(state_his, observation_new)
        state = state_new
        running_reward += reward

    # update
        tot_steps += 1

        if done:
            print(running_reward)
            call(['pkill', '-9', 'fceux'])
            env = gym.make('ppaquette/SuperMarioBros-1-1-Tiles-v0')
            obs = env.reset()
            obs = obs[np.newaxis, :]  # (1,13,16)

            break
print('end')
call(['pkill', '-9', 'fceux'])



# write_log(LR,tot_steps)
