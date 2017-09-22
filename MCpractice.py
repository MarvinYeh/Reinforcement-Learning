
# from keras.models import Sequential      # One layer after the other
# from keras.layers import Dense # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
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
EPISILON = 0.4                            # Probability of doing a random move
gamma = 0.9                                # Discounted future reward. How much we care about steps further in time
mb_size = 64                              # Learning minibatch size
n_feature = env.observation_space.shape[0]
n_action = env.action_space.n
tot_steps = 1
MEMORY_SIZE = 3000
# LOGDIR =  os.path.join(os.getcwd(),'log_mc')
 # D = deque() # initialize memory
D2 = deque(maxlen=3000) # initialize memory


with tf.variable_scope('eval-net'):
    tfx = tf.placeholder(tf.float32,[None,8],name='tfx')
    tfy = tf.placeholder(tf.float32,[None,3],name='tfy')
    hid1 = tf.layers.dense(tfx,10,activation=tf.nn.relu)
    out = tf.layers.dense(hid1,3,activation=None)
    loss = tf.losses.mean_squared_error(tfy,out)
    train_op = tf.train.AdamOptimizer(0.001).minimize(loss)


# target
with tf.variable_scope('target-net'):
    tfx_t = tf.placeholder(tf.float32,[None,8])
    tfy_t = tf.placeholder(tf.float32,[None,3])
    hid1_t = tf.layers.dense(tfx_t,10,activation=tf.nn.relu)
    out_t = tf.layers.dense(hid1_t,3,activation=None)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()


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

def store_memory(state, action, reward, done, state_new, memory,tot_steps):
    m_ind = tot_steps % MEMORY_SIZE
    # memory[m_ind,:8] = state
    # memory[m_ind,8] = action
    # memory[m_ind,9] = reward
    # memory[m_ind,10]= done
    # memory[m_ind,11:] = state_new

    return memory


for epi in range(10):

    # obs = np.expand_dims(env.reset(), axis = 0)  # format for keras (2,)->(1,2)
    obs = env.reset()
    state = np.expand_dims(np.hstack((obs,obs,obs,obs)),axis=0)
    # initialize memory buffer
    memory = np.zeros((MEMORY_SIZE, state.shape[1] * 2 + 2 + 1))  # state*2 + act+ rwd + done

    while True:
    # for i in range(30000):
        env.render()

        action=choose_action(EPISILON,state)

        observation_new, reward, done, info = env.step(action)

        # define new reward for MC
        # reward = abs(observation_new[0] - (-0.5))
        # reward = observation_new[0] - (-0.5)
        # if done:
        #     reward = 100

        # obs_new = np.expand_dims(observation_new ,axis=0)  # format
        state_new = get_state(observation_new,state) # state_new.shape= (1,8)
        # memory = store_memory(state, action, reward, done, state_new, memory,tot_steps)
        D2.append([state, action, reward, done, state_new])
        # D2.append((obs[0,0],obs[0,1],action,reward,obs_new[0,0],obs_new[0,1]))
        state = state_new
        tot_steps += 1
        print(tot_steps,'| win: ',win_count,'| epsilon: ', EPISILON)

        #learning: sample random minibatch
        if tot_steps >500:

            # tf model
            # minibatch = np.asarray(random.sample(D2, mb_size))

            # inputs = minibatch[:,:2]
            # targets = sess.run(out,{tfx:inputs})
            # action = minibatch[:,2].astype(int)
            # rewards = minibatch[:,3]
            # Q_sa = sess.run(out_t,{tfx_t:minibatch[:,4:]})
            #





            # deque
            minibatch = np.asarray(random.sample(D2, mb_size))
            inputs = np.vstack(minibatch[:,0])
            targets = sess.run(out,{tfx:inputs})
            action = minibatch[:,1].astype(int)
            rewards = minibatch[:,2]
            input_next = np.vstack(minibatch[:,4])
            Q_sa = sess.run(out_t,{tfx_t:input_next})
            Q_ind = np.argmax(targets,axis=1)
            # print(Q_ind)



            batch_index = np.arange(mb_size,dtype=int)
            targets[batch_index,action] = rewards[batch_index] + gamma * targets[batch_index ,Q_ind]
            # train model
            # loss, acc = model.train_on_batch(inputs, targets)
            _,loss_ = sess.run([train_op,loss],{tfx:inputs,tfy:targets})

            cost_history.append(loss_)

            if tot_steps % 300 ==0:
                # model_target.set_weights(model.get_weights())
                # # if epsilon > 0:
                # #     epsilon -= 0.01
                # print('update parameter')
                #
                #tf model
                e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval-net')
                t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target-net')
                sess.run([tf.assign(t,e) for t,e in zip(t_params,e_params)])
                print('update parameter')
        if done:
            # print("Win!")
            win = tot_steps
            print("Win",win)
            train_summary.append(tot_steps)
            win_count += 1
            break
        if tot_steps % 2000 ==0 :
            saver.save(sess, os.path.join("./MC"), tot_steps)
            EPISILON -= 0.005


train_step = [j-i for i,j in zip(train_summary[:-1],train_summary[1:])]
plt.plot(np.arange(len(cost_history)),cost_history)
plt.ylabel('Cost')
plt.xlabel('training steps')
plt.show()

np.savetxt('win summary.txt',train_summary,delimiter=',')

print(train_summary)
print(train_step)
