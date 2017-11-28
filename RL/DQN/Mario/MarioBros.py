
import matplotlib.pyplot as plt
from collections import deque            # For storing moves
import numpy as np
import gym                                # To train our network
import random     # For sampling batches from the observations
import tensorflow as tf
import os


env = gym.make('SuperMarioBros-1-1-v0')
# env.no_render = False
# env = env.unwrapped
#
# print(env.action_space)             #MultiDiscrete(6)
# print(env.observation_space)
# print(env.observation_space.high)   # Box(224,256,3 )
# print(env.observation_space.low)

# initail setting#
epsilon = 0.5                            # Probability of doing a random move
gamma = 0.9                                # Discounted future reward. How much we care about steps further in time
mb_size = 32                              # Learning minibatch size
# n_feature = env.observation_space.shape[0]
# n_action = env.action_space.n
tot_steps = 1
D = deque() # initialize memory
MEMORY_SIZE = 3000
BATCH_SIZE = 50
REPLACE_ITER = 500
LOGDIR =  os.path.join(os.getcwd(),'log_mario')


###function###
def sig(x):
    return 1/(1+np.exp(-x))

# CNN-1
with tf.variable_scope('eval-net'):

    tfx = tf.placeholder(tf.float32,[None,224,256,3])/255
    image = tf.reshape(tfx,[-1,224,256,3])
    tfy = tf.placeholder(tf.float32,[None,6])
    conv1 = tf.layers.conv2d(inputs=image, filters=32, kernel_size=5,activation=tf.nn.relu) #( ,220,252,32)
    conv2 = tf.layers.conv2d(conv1, 16,5,1,activation=tf.nn.relu) #(216,248,16)
    flat = tf.reshape(conv2,[-1,216*248*16])
    hidden1 = tf.layers.dense(flat,100,activation=tf.nn.relu)
    output = tf.layers.dense(hidden1,6,activation=None)
    loss = tf.reduce_mean(tf.squared_difference(output,tfy))




#CNN-target
with tf.variable_scope('target-net'):
    tfx_t = tf.placeholder(tf.float32,[None,224,256,3],name='tfx_t')/255
    image_t = tf.reshape(tfx_t,[-1,224,256,3],name='tfy_t')
    tfy_t = tf.placeholder(tf.float32,[None,6])
    conv1_t = tf.layers.conv2d(inputs=image_t, filters=32, kernel_size=5,activation=tf.nn.relu) #( ,220,252,32)
    conv2_t = tf.layers.conv2d(conv1_t, 16,5,1,activation=tf.nn.relu) #(216,248,64)
    flat_t = tf.reshape(conv2_t,[-1,216*248*16])
    hidden1_t = tf.layers.dense(flat_t,100,activation=tf.nn.relu)
    output_t = tf.layers.dense(hidden1_t,6,activation=None)


# train_op = tf.train.RMSPropOptimizer(0.01).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# writer = tf.summary.FileWriter('./log', sess.graph)
# merge_op = tf.summary.merge_all()
saver = tf.train.Saver()


# initialize memory
tot_steps = 1
observation_history = np.empty((MEMORY_SIZE,224,256,3))
observation_new_history = np.empty((MEMORY_SIZE,224,256,3))
action_history = np.empty((MEMORY_SIZE,6))
reward_history = np.empty((MEMORY_SIZE,1))
done_history = np.empty((MEMORY_SIZE,1))

observation = env.reset()
loss_history = np.zeros((5000,1))



while True:
# for step in range(200):




# state = sess.run(output,{tfx:observation})
    state = observation[np.newaxis,:]

    # action = [0,0,0,1,0,0];
    # observation_new, reward, done, info = env.step(action)

    if np.random.rand() <= epsilon:  # random
        action = env.action_space.sample()
    else:  # learned action
        action = sess.run(output,{tfx:state})
        action = sig(action).reshape((6,)).tolist()

    observation_new, reward, done, info = env.step(action)  #observation (224,256,3), reward float, info:https://github.com/ppaquette/gym-super-mario/blob/master/ppaquette_gym_super_mario/README.txt

    print(done,tot_steps)

    state_new = observation_new[np.newaxis,:]

    # save s,a,r,s_
    # D.append((observation,action ,reward, done, observation_new)) # obs_new,reward,done
    index = tot_steps % MEMORY_SIZE
    observation_history[index] = state
    observation_new_history[index] = state_new
    action_history[index] = action
    reward_history[index] = reward
    done_history[index] = done

    # update
    tot_steps += 1
    observation = observation_new


# training
    if tot_steps >= 500:

        # random select
        if tot_steps>3000:
            rnd_idx = np.random.randint(0,3000,BATCH_SIZE)
        else:
            rnd_idx = np.random.randint(0,tot_steps,BATCH_SIZE)
        # replace target new
        if tot_steps % REPLACE_ITER ==0:
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval-net')
            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target-net')
            sess.run([tf.assign(t,e) for t,e in zip(t_params,e_params)])
            print('replace target parameter')

        input = observation_history[rnd_idx]
        act = action_history[rnd_idx]
        rwd = reward_history[rnd_idx]
        obs_new = observation_new_history[rnd_idx]
        target = sess.run(output,{tfx:input})
        finish = done_history[rnd_idx]
        Q_sa = sess.run(output_t,{tfx_t:obs_new})

        batch_index = np.arange(BATCH_SIZE, dtype=int)
        act_idx=np.where(act[batch_index]==1)

        for j in range(act_idx[0].shape[0]):
            target[act_idx[0][j],act_idx[1][j]] = rwd[act_idx[0][j]]+ gamma*np.argmax(Q_sa[act_idx[0][j]])
        _, loss_ = sess.run([train_op,loss],{tfx:input,tfy:target})
        loss_history[tot_steps-499] = loss_.mean()
        # writer.add_summary(result,tot_steps)

        if tot_steps % 1000 ==0 :
            saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), tot_steps)
            epsilon -= 0.001

saver.save(sess,'params_mario', write_meta_graph=False)
plt.plot(loss_history)

plt.show()






#
# # Learning
#
#
# for step in range(200):
#     _,l = sess.run([train_op,loss],{tfx:x,tfy:y})
#     if step % 20 == 0:
#         print('Step:',step,'|Loss:%.4f' % l)
