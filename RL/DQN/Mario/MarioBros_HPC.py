
import matplotlib.pyplot as plt
from collections import deque            # For storing moves
import numpy as np
import random     # For sampling batches from the observations
import tensorflow as tf
import os
from subprocess import call
import pickle

# initail setting#
EPISILON = 1                            # Probability of doing a random move
gamma = 0.9                                # Discounted future reward. How much we care about steps further in time
mb_size = 64                              # Learning minibatch size
tot_steps = 1
D = deque() # initialize memory
MEMORY_SIZE = 10000
BATCH_SIZE = 50
REPLACE_ITER = 1000
LR= 0.001
# LOGDIR =  os.path.join(os.getcwd(),'log_mario')
STATE_SIZE = 6
EPOCH = 10



#
###function###

# # CNN-1
with tf.variable_scope('eval-net'):

    tfx = tf.placeholder(tf.float32,[None,6,13,16])
    flat = tf.reshape(tfx,[-1, 6*13*16])
    tfy = tf.placeholder(tf.float32,[None,6])
    hidden1 = tf.layers.dense(flat,128,activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1,64 , activation=tf.nn.relu)
    output = tf.layers.dense(hidden2,6,activation=None)
    loss = tf.reduce_mean(tf.squared_difference(output,tfy))

with tf.variable_scope('target-net'):

    tfx_t = tf.placeholder(tf.float32,[None,6,13,16])
    flat_t = tf.reshape(tfx_t,[-1, 6*13*16])
    tfy_t = tf.placeholder(tf.float32,[None,6])
    hidden1_t = tf.layers.dense(flat_t,128,activation=tf.nn.relu)
    hidden2_t = tf.layers.dense(hidden1_t,64 , activation=tf.nn.relu)
    output_t = tf.layers.dense(hidden2_t,6,activation=None)

train_op = tf.train.AdamOptimizer(LR).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()


STARTING_STEP = 0

# check if there is file in ./mario-checkpoint


# fname = './mario-logs/Mario-'+ str(STARTING_STEP)
# saver.restore(sess,fname)

# assign the same parameter
e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval-net')
t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target-net')
sess.run([tf.assign(t,e) for t,e in zip(t_params,e_params)])


# # writer = tf.summary.FileWriter('./log', sess.graph)
# # merge_op = tf.summary.merge_all()
saver = tf.train.Saver()



#
#
# # initialize memory
tot_steps = 1

train_summary = []
cost_history = []
win_count = 0
max_reward = 0



# while True:
for epoch in range(EPOCH):

    for r in range(len(os.listdir('./mario-logs'))):

        fname = './mario-logs/'+ os.listdir('./mario-logs')[r]
        print(fname)
        D = pickle.load(open(fname,'rb'))
        print('read '+fname)
        for _ in range(10000):

        # replace target new
            if tot_steps % REPLACE_ITER ==0:
                e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval-net')
                t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target-net')
                sess.run([tf.assign(t,e) for t,e in zip(t_params,e_params)])
                # print('replace target parameter')
            minibatch = np.asarray(random.sample(D, mb_size))
            inputs = np.vstack(minibatch[:, 0])
            targets = sess.run(output, {tfx: inputs})
            actions = np.vstack(minibatch[:, 1])
            rewards = minibatch[:, 2]
            input_next = np.vstack(minibatch[:, 3])
            Q_sa = sess.run(output_t, {tfx_t: input_next})
            Q_ind = np.argmax(targets, axis=1)

            # batch_index = np.arange(mb_size, dtype=int)
            idx_x, idx_y = np.where(actions==1)

            # double dqn
            targets[idx_x,idx_y] = rewards[idx_x] + gamma * Q_sa[idx_x, Q_ind[idx_x]]
            # Natural dqn
            # targets[batch_index, action] = rewards[batch_index] + gamma * np.max(Q_sa,axis=1)
            _, loss_ = sess.run([train_op, loss], {tfx: inputs, tfy: targets})

            tot_steps += 1

    saver.save(sess, os.path.join("./mario-checkpoint/Mario"), epoch)
    print('Epoch:{}'.format(epoch))
# write_log(LR,tot_steps)
