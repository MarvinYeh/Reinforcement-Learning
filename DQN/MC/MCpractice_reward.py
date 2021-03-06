# import matplotlib.pyplot as plt
from collections import deque  # For storing moves
import numpy as np
import gym  # To train our network
import random  # For sampling batches from the observations
import tensorflow as tf
import os
from pathlib import Path
import pickle

# env = gym.make('SuperMarioBros-1-1-v0')
env = gym.make('MountainCar-v0')  # Choose game (any in the gym should work)
env = env.unwrapped

# print parameter
# print(env.action_space)
# print(env.observation_space)  #Box(2,)
# print(env.observation_space.high) # [0.6 0.07]
# print(env.observation_space.low)

# initail setting#
EPISILON = 1  # Probability of doing a random move
gamma = 0.99  # Discounted future reward. How much we care about steps further in time
mb_size = 64  # Learning minibatch size
n_feature = env.observation_space.shape[0]
n_action = env.action_space.n
tot_steps = 1
MEMORY_SIZE = 100000

LR = 0.01


D2 = deque(maxlen=MEMORY_SIZE)  # initialize memory

with tf.variable_scope('eval-net'):
    tfx = tf.placeholder(tf.float32, [None, 2], name='tfx')
    tfy = tf.placeholder(tf.float32, [None, 3], name='tfy')
    hid1 = tf.layers.dense(tfx, 20, activation=tf.nn.relu)
    out = tf.layers.dense(hid1, 3, activation=None)
    # loss = tf.losses.mean_squared_error(labels=tfy,predictions=out)
    loss = tf.reduce_mean(tf.squared_difference(out, tfy))
    train_op = tf.train.RMSPropOptimizer(LR).minimize(loss)

# target
with tf.variable_scope('target-net'):
    tfx_t = tf.placeholder(tf.float32, [None, 2])
    tfy_t = tf.placeholder(tf.float32, [None, 3])
    hid1_t = tf.layers.dense(tfx_t, 20, activation=tf.nn.relu)
    out_t = tf.layers.dense(hid1_t, 3, activation=None)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval-net')
t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target-net')
sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])


train_summary = []
cost_history = []
win_count = 0


###functions###

def choose_action(epsilon, state):
    if np.random.rand() <= epsilon:  # random
        action = np.random.randint(0, n_action)
        action = env.action_space.sample()
    else:  # learned action
        Q = sess.run(out, {tfx: state})
        action = np.argmax(Q)
    return action


def write_log(LR, tot_steps):
    with open('./MC-reward-log.txt', 'a') as file:
        file.write('LR:{}- Steps:{}\n'.format(LR,tot_steps))


s1 = np.reshape([-0.15955113, 0.], (1, 2))    #start
s2 = np.reshape([0.83600049, 0.27574312], (1, 2))
s3 = np.reshape([0.85796947, 0.28245832], (1, 2))
s4 = np.reshape([0.88062271, 0.29125591], (1, 2)) #end
Q_his =deque()

while True:
# for epi in range(200):

    obs = env.reset()
    obs = np.expand_dims(obs, axis=0)

    while True:
    # for i in range(1000):
        #     env.render()

        action = choose_action(EPISILON, obs)

        observation_new, reward, done, info = env.step(action)

        # if done: reward = 10

        obs_new = np.expand_dims(observation_new, axis=0)  # format

        D2.append([obs, action, reward, done, obs_new])
        # D2.append((obs[0,0],obs[0,1],action,reward,obs_new[0,0],obs_new[0,1]))
        obs = obs_new
        tot_steps += 1


        # learning: sample random minibatch
        if tot_steps > 1000:
            if tot_steps % 1000 == 0:
                Q_his.append([s1, s2, s3, s4])
            if tot_steps % 10000 == 0:
                print(sess.run(out, {tfx: s1}),'|step: {}'.format(tot_steps))
            if tot_steps % 500 == 0:
                # tf model
                e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval-net')
                t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target-net')
                sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

            # deque
            minibatch = np.asarray(random.sample(D2, mb_size))
            inputs = np.vstack(minibatch[:, 0])
            targets = sess.run(out, {tfx: inputs})
            action = minibatch[:, 1].astype(int)
            rewards = minibatch[:, 2]
            input_next = np.vstack(minibatch[:, 4])
            Q_sa = sess.run(out_t, {tfx_t: input_next})
            Q_ind = np.argmax(targets, axis=1)
            # print(Q_ind)



            batch_index = np.arange(mb_size, dtype=int)
            # double dqn
            targets[batch_index, action] = rewards[batch_index] + gamma * Q_sa[batch_index, Q_ind]
            # Natural dqn
            # targets[batch_index, action] = rewards[batch_index] + gamma * np.max(Q_sa,axis=1)
            _, loss_ = sess.run([train_op, loss], {tfx: inputs, tfy: targets})

            cost_history.append(np.mean(loss_))
            EPISILON -= 0.00005
            if EPISILON < 0.01:
                EPISILON = 0.01


                # print('update parameter|',tot_steps)
        if done:
            win = tot_steps
            win_count += 1
            # print(tot_steps, '| win: ', win_count, '| epsilon: ', EPISILON,'|Q:' ,Q_sa[0])
            train_summary.append(tot_steps)

            break
    if tot_steps >=1000*1000:
        break

saver.save(sess, os.path.join("./MC_reward"), tot_steps)
write_log(LR,tot_steps)
mem_name = './Q_his-lr-{}'.format(LR) + '.p'
pickle.dump(Q_his, open(mem_name, 'wb'))

test_input = np.zeros((1, 2))
print(sess.run(out, {tfx: test_input}))
#
print(out)
# # train_step = [j-i for i,j in zip(train_summary[:-1],train_summary[1:])]
# plt.plot(np.arange(len(cost_history)), cost_history)
# plt.show()
# plt.plot(np.arange(len(train_summary)), train_summary)
# plt.ylabel('steps')
# plt.xlabel('episodes')
# plt.show()

# np.savetxt('win summary.txt',train_summary,delimiter=',')

# print(train_summary)
# print(train_step)
