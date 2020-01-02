'''
code modified from https://github.com/awjuliani/DeepRL-Agents
'''

# In[1]:


from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os
from mouselab import MouselabEnv
from distributions import Categorical, Normal
import pandas as pd
import shutil


# In[2]:


def make_envs2(cost=1, n=100):
    def reward(depth):
        if depth == 1:
            return Categorical([-4, -2, 2, 4])
        elif depth == 2:
            return Categorical([-8, -4, 4, 8])
        elif depth == 3:
            return Categorical([-48, -24, 24, 48])

    branching = [3, 1, 2]
    return [MouselabEnv.new_symmetric(branching, reward, seed=i, cost=cost, sample_term_reward=False, env_type='new')
            for i in range(n)]


env_array = make_envs2(n=500)


# In[3]:


def processState(instate):
    mask = np.ones(len(instate))
    state = np.ones(len(instate))
    for i in range(0, len(instate)):
        if hasattr(instate[i], 'sample'):
            mask[i] = -1
            state[i] = 0
        else:
            mask[i] = 1
            state[i] = instate[i]
    state = state[1:]
    mask = mask[1:] * 60
    out = np.stack((state, mask), axis=1).flatten()

    return out


# In[4]:


class Qnetwork():
    def __init__(self, h_size, n_obs, n_actions):

        self.scalarInput = tf.placeholder(
            shape=[None, n_obs * 2], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 1, n_obs, 2])

        self.conv1 = slim.conv2d(inputs=self.imageIn, num_outputs=32, kernel_size=[
                                 1, 4], stride=[1, 2], padding='VALID', biases_initializer=None)

        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=64, kernel_size=[
                                 1, 3], stride=[1, 2], padding='VALID', biases_initializer=None)

        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=h_size, kernel_size=[
                                 1, 2], stride=[1, 1], padding='VALID', biases_initializer=None)

        # We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC, self.streamVC = tf.split(self.conv3, 2, 3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size // 2, n_actions]))
        self.VW = tf.Variable(xavier_init([h_size // 2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + \
            tf.subtract(self.Advantage, tf.reduce_mean(
                self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(
            self.actions, n_actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(
            self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)


# In[5]:


class experience_buffer():
    def __init__(self, buffer_size=8000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) -
                        self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


# In[6]:


def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars // 2]):
        op_holder.append(tfVars[idx + total_vars // 2].assign(
            (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
    return op_holder


def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)


# ### Training the network

# In[7]:


batch_size = 32  # How many experiences to use for each training step.
update_freq = 2  # How often to perform a training step.
y = .99  # Discount factor on the target Q-values
startE = 1  # Starting chance of random action
endE = 0.1  # Final chance of random action
# How many episodes of game environment to train network with.
num_episodes = 25000
# How many steps of training to reduce startE to endE.
annealing_steps = num_episodes * 2
# How many steps of random actions before training begins.
pre_train_steps = num_episodes
load_model = False  # Whether to load a saved model.

path = "./models/temp/"
# The size of the final convolutional layer before splitting it into Advantage and Value streams.
h_size = 256
tau = 0.001  # Rate to update target network toward primary network
env = env_array[np.random.randint(0, 500)]
n_obs = env.n_obs
n_actions = env.n_actions
max_epLength = env.n_actions  # The max allowed length of our episode.


# In[ ]:


tf.reset_default_graph()
mainQN = Qnetwork(h_size, n_obs, n_actions)
targetQN = Qnetwork(h_size, n_obs, n_actions)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables, tau)

myBuffer = experience_buffer()

# Set the rate of random action decrease.
e = startE
stepDrop = (startE - endE) / annealing_steps

# create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

try:
    shutil.rmtree('./tfb')
except OSError:
    print ("no dir")


files = glob.glob(path+'*')
for f in files:
    os.remove(f)

summary_writer = tf.summary.FileWriter("tfb")

with tf.Session() as sess:
    sess.run(init)
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    for i in range(num_episodes):

        episodeBuffer = experience_buffer()
        # Reset environment and get first new observation
        env = env_array[np.random.randint(0, 500)]

        s = env.reset()
        s = processState(s)
        d = False
        rAll = 0
        j = 0
        # The Q-Network
        while j < max_epLength:
            j += 1

            # Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0, n_actions)
            else:
                a = sess.run(mainQN.predict, feed_dict={
                             mainQN.scalarInput: [s]})[0]

            s1, r, d, obs = env.step(a)

            s1 = processState(s1)
            total_steps += 1
            # Save the experience to our episode buffer.
            episodeBuffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))

            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop

                if total_steps % (update_freq) == 0:

                    # Get a random batch of experiences.
                    trainBatch = myBuffer.sample(batch_size)
                    # Below we perform the Double-DQN update to the target Q-values

                    Q1 = sess.run(mainQN.predict, feed_dict={
                                  mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    Q2 = sess.run(targetQN.Qout, feed_dict={
                                  targetQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    end_multiplier = -(trainBatch[:, 4] - 1)
                    doubleQ = Q2[range(batch_size), Q1]
                    targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                    # Update the network with our target values.
                    _ = sess.run(mainQN.updateModel,                         feed_dict={mainQN.scalarInput: np.vstack(
                        trainBatch[:, 0]), mainQN.targetQ: targetQ, mainQN.actions: trainBatch[:, 1]})

                    # Update the target network toward the primary network.
                    updateTarget(targetOps, sess)



            rAll += r
            print(i, a, rAll, e)
            s = s1
            if d == True:

                break

        if i % 500 == 0:
            print(i,e)

            mean_reward = np.mean(rList[-500:])
            summary = tf.Summary()
            summary.value.add(tag='Perf/Reward',simple_value=float(mean_reward))
            summary_writer.add_summary(summary, i)
            summary_writer.flush()

        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)
        rList.append(rAll)


        if i>7000 and i%2000==0:
            saver.save(sess, path+str(i)+'.cptk')






# In[15]:


tf.reset_default_graph()
mainQN = Qnetwork(h_size, n_obs, n_actions)
targetQN = Qnetwork(h_size, n_obs, n_actions)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables, tau)

myBuffer = experience_buffer()
test_epLength = n_actions
df = pd.DataFrame(columns=['i', 'return', 'actions', 'ground_truth'])
resum = 0
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(path)
    saver.restore(sess,ckpt.model_checkpoint_path)

    for i in range(500):
        env = env_array[i]
        sP = env.reset()
        s = processState(sP)
        actions = []
        d = False
        rAll = 0
        j = 0

        while j < test_epLength:
            j += 1
            a = sess.run(mainQN.predict, feed_dict={
                         mainQN.scalarInput: [s]})[0]

            s1P, r, d, obs = env.step(a)
            s1 = processState(s1P)

            if(r == -50):
                r = 0

                s = s1
                print("hi")
                continue

            rAll += r
            actions.append(a)
            s = s1
            sP = s1P

            if d == True:
                break

        if(d == False):
            a = 0
            actions.append(a)
            s1, r, d, obs = env.step(a)
            rAll += r

        df.loc[i] = [i, rAll, actions, env.ground_truth]
        print(i, rAll, actions)
        resum += rAll

#df.to_csv('dqn.csv')
print(resum / 500)



# In[15]:


# In[15]:
