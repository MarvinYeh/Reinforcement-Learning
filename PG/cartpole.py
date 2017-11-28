import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt


try:
    xrange = xrange
except:
    xrange = range

env = gym.make('CartPole-v0')
# env = gym.make('MountainCar-v0')
env.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

gamma = 0.99

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class agent():
    def __init__(self, lr, s_size,a_size,h_size=8):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        self.chosen_action = tf.argmax(self.output,1)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network. Both of length equal to the number of steps in the episode
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.oh_action = tf.one_hot(self.action_holder, depth=2, axis=-1)
        
        # Make a long vector where values index into the action taken at that step
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder

        # Slice the output vector by the indices to return only those softmax probs pertaining to the actions taken. This so when we 
        # backpropagate only the relevant weights are updated
        # self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)
        self.responsible_outputs = tf.reduce_sum(tf.multiply(self.output, self.oh_action), axis=1)

        self.loss = tf.reduce_mean(-tf.log(self.responsible_outputs)*self.reward_holder)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)
        
            
tf.reset_default_graph() #Clear the Tensorflow graph.

myAgent = agent(lr=1e-2,s_size=4,a_size=2) #Load the agent.
total_episodes = 1 #Set total number of episodes to train agent on.
max_steps = 501  # 999
update_frequency = 1

init = tf.global_variables_initializer()

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_reward = []
    total_length = []
    ep_history = []
        
    while i < total_episodes:
        s = env.reset()
        running_reward = 0
        
        for j in range(max_steps):

            # if i % 1000 == 0:
                # env.render()

            #Probabilistically pick an action given our network outputs.
            a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[s]})
            a = np.random.choice(a_dist[0], p=a_dist[0]) #returns either the first or second value of a_dist with prob equal to that value
            a = np.argmax(a_dist == a) # returns the index where statement is true ie the chosen action

            s1,r,d,_ = env.step(a) #Get our reward for taking an action
            ep_history.append([s,a,r,s1])
            s = s1
            running_reward += r

          


            if d == True:
                #Update the network.
                ep_history = np.array(ep_history)
                ep_history[:,2] = discount_rewards(ep_history[:,2])
                feed_dict={myAgent.reward_holder:ep_history[:,2],
                        myAgent.action_holder:ep_history[:,1],myAgent.state_in:np.vstack(ep_history[:,0])}

                


                if i % update_frequency == 0 and i != 0:
                    _ = sess.run(myAgent.optimizer, feed_dict=feed_dict)

                    # print(ep_history[:,2])
      
                total_reward.append(running_reward)
                total_length.append(j)
                ep_history = []




                break



        
            #Update our running tally of scores.
        if i % 100 == 0:
            print('episode %d | total reward: %d' % (i,np.mean(total_reward[-100:])))
        
        i += 1

