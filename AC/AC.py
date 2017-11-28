import numpy as np
import tensorflow as tf
import gym

# env =gym.make('Cartpole-v0')


# GAMMA = 0.9     # reward discount in TD error
# LR_A = 0.001    # learning rate for actor
# LR_C = 0.01     # learning rate for critic
# N_F = env.observation_space.shape[0]
# N_A = env.action_space.n
#

class Actor(object):

    def __init__(self,sess,n_f, n_a, lr = 0.001):
        self.sess = sess
        self.tfx = tf.placeholder(tf.float32,[None, n_f])
        self.tfy = tf.placeholder(tf.int32, None)
        self.td = tf.placeholder(tf.float32, None)

        with tf.variable_scope('Actor'):
            hid1 = tf.layers.dense(self.tfx,20,activation=tf.nn.relu)
            self.out = tf.layers.dense(hid1,n_a,activation=tf.nn.softmax) # output action probability
            log_prob = tf.log(self.out[0,self.tfy])
            self.exp_v = tf.reduce_mean(log_prob * self.td)
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)

    def learn(self,s,a,td):
        s = s[np.newaxis,:]
        _ = self.sess.run(self.train_op,{self.tfx:s,self.tfy:a,self.td:td})

    def choose_action(self,s):
        s = s[np.newaxis,:]
        out_ = self.sess.run(self.out,{self.tfx:s})
        return np.random.choice(np.arange(out_.shape[1]),p=out_.ravel())


class Critic(object):

    def __init__(self, sess, n_f, lr=0.01):
        self.sess = sess
        self.tfx = tf.placeholder(tf.float32,[None,n_f])
        self.tfv_ = tf.placeholder(tf.float32,[None,n_f])
        self.tfr = tf.placeholder(tf.float32, None)

        with tf.variable_scope('Critic'):
            hid1 = tf.layers.dense(self.tfx,20,activation=tf.nn.relu)
            self.v = tf.layers.dense(hid1,1,activation=None) # output v for this s
            self.td = self.tfr+ GAMMA * self.v -self.tfv_
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.td)


    def learn(self,s,r,s_):
        s,s_ = s[np.newaxis,:],  s_[np.newaxis,:]
        v_= self.sess.run(self.v,{self.tfx:s_})
        _, td = self.sess.run([self.train_op,self.td],{self.tfx:s,self.tfv_:v_, self.tfr:r})
        return td

if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    GAMMA = 0.9  # reward discount in TD error
    LR_A = 0.001  # learning rate for actor
    LR_C = 0.01  # learning rate for critic
    N_F = env.observation_space.shape[0]
    N_A = env.action_space.n
    rwd_his = []


    sess = tf.Session()
    act = Actor(sess,n_f=N_F,n_a=N_A,lr=LR_A)
    crit = Critic(sess,n_f=N_F,lr=LR_C)
    sess.run(tf.global_variables_initializer())

    for epi in range(100):
        s = env.reset()
        running_reward = 0
        while True:
            a = act.choose_action(s)
            s_,r,done,info = env.step(a)
            # memory.append([s,a,r,s_])
            td = crit.learn(s,r,s_)
            act.learn(s,a,td)

            s = s_
            running_reward += r
            if done:
                rwd_his.append(running_reward)
                break
        if epi % 10 ==0:
            print('epi:{}|rwds:{}'.format(epi,np.mean(running_reward)))
            rwd_his = []
