'''
code modified from https://github.com/awjuliani/DeepRL-Agents
'''


# In[1]:
from helper import *
import pandas as pd
from agents import Agent
from contexttimer import Timer
from exact import solve
from policies import FunctionPolicy
from distributions import Categorical, Normal
from mouselab import MouselabEnv
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
import shutil
import csv
import glob
import itertools
import tensorflow.contrib.slim as slim
from generate_environments import *

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
    def __init__(self, h_size, rnn_cell, myScope, n_obs, n_actions, hidden_size):

        self.scalarInput = tf.placeholder(
            shape=[None, n_obs * 2], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 1, n_obs, 2])

        self.conv1 = slim.conv2d(inputs=self.imageIn, num_outputs=32, kernel_size=[1, 4], stride=[
                                 1, 2], padding='VALID', biases_initializer=None, scope=myScope + '_conv1')

        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=90, kernel_size=[1, 3], stride=[
                                 1, 2], padding='VALID', biases_initializer=None, scope=myScope + '_conv2')

        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=h_size, kernel_size=[1, 2], stride=[
                                 1, 2], padding='VALID', biases_initializer=None, scope=myScope + '_conv3')


        self.prev_rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.prev_actions_onehot = tf.one_hot(self.prev_actions, n_actions+1, dtype=tf.float32)

        self.hidden = tf.concat([slim.flatten(self.conv3)], 1)

        self.trainLength = tf.placeholder(dtype=tf.int32)

        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])

        self.convFlat = tf.reshape(
            self.hidden, [self.batch_size, self.trainLength, hidden_size])

        self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(
            inputs=self.convFlat, cell=rnn_cell, dtype=tf.float32, initial_state=self.state_in, scope=myScope + '_rnn')
        self.rnn = tf.reshape(self.rnn, shape=[-1, hidden_size])
        # The output from the recurrent player is then split into separate Value and Advantage streams
        self.streamA, self.streamV = tf.split(self.rnn, 2, 1)
        self.AW = tf.Variable(tf.random_normal([hidden_size // 2, n_actions]))
        self.VW = tf.Variable(tf.random_normal([hidden_size // 2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        self.salience = tf.gradients(self.Advantage, self.imageIn)
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

        self.maskA = tf.zeros([self.batch_size, self.trainLength // 2])

        self.maskB = tf.ones(
            [self.batch_size, self.trainLength - self.trainLength // 2])
        self.mask = tf.concat([self.maskA, self.maskB], 1)
        self.mask = tf.reshape(self.mask, [-1])
        self.loss = tf.reduce_mean(self.td_error * self.mask)

        self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.updateModel = self.trainer.minimize(self.loss)


# In[5]:


class experience_buffer():
    def __init__(self, buffer_size=4000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size):
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampledTraces = []
        trace_len = 0
        for episode in sampled_episodes:
            sampledTraces.append(episode)
            trace_len = len(episode)
        sampledTraces = np.array(sampledTraces)

        ret = np.reshape(sampledTraces, [batch_size * trace_len, 5])
        return ret, trace_len


# In[6]:


# Setting the training parameters
batch_size = 1  # How many experience traces to use for each training step.
update_freq = 5  # How often to perform a training step.
y = .99  # Discount factor on the target Q-values
startE = 1  # Starting chance of random action
endE = 0.15  # Final chance of random action
num_episodes = 20000  # How many steps of training to reduce startE to endE.
annealing_steps = 4000 * 10
pre_train_steps = 2000 * 10
path = "./models/temp/"  # The path to save our model to.
learning_rate = 1e-4
time_per_step = 1  # Length of each step used in gif creation
summaryLength = 100  # Number of epidoes to periodically save for analysis
tau = 0.001
h_size = 300
hidden_size = h_size


# In[7]:

def train_dqnrnn(env_array):

    env = env_array[0]
    n_obs = env.n_obs
    n_actions = env.n_actions
    max_epLength = n_actions
    num_envs = len(env_array)

    tf.reset_default_graph()
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
    cellT = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
    mainQN = Qnetwork(h_size, cell, 'main', n_obs, n_actions, hidden_size)
    targetQN = Qnetwork(h_size, cellT, 'target', n_obs, n_actions, hidden_size)

    init = tf.global_variables_initializer()
    trainables = tf.trainable_variables()
    targetOps = updateTargetGraph(trainables, tau)
    myBuffer = experience_buffer()
    saver = tf.train.Saver()

    # Set the rate of random action decrease.
    e = startE
    stepDrop = (startE - endE) / annealing_steps

    # create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    total_steps = 0
    e_increase=True

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

        # Set the target network to be equal to the primary network.
        updateTarget(targetOps, sess)
        for i in range(num_episodes):
            episodeBuffer = []
            # Reset environment and get first new observation
            env = env_array[np.random.randint(0, num_envs)]
            sP = env.reset()
            s = processState(sP)
            d = False
            r = 0
            a = 13
            rAll = 0
            j = 0
            random_ac = random.sample(range(0,n_actions), n_actions)
            # Reset the recurrent layer's hidden state
            state = (np.zeros([1, hidden_size]), np.zeros([1, hidden_size]))
            # The Q-Network
            while j < max_epLength:
                j += 1
                # Choose an action by greedily (with e chance of random action) from the Q-network
                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    state1 = sess.run(mainQN.rnn_state,
                                      feed_dict={mainQN.scalarInput: [s], mainQN.trainLength: 1, mainQN.state_in: state, mainQN.batch_size: 1, mainQN.prev_rewards: [[r]], mainQN.prev_actions: [a]})
                    a = random_ac[0]
                    random_ac = random_ac[1:]

                else:
                    a, state1 = sess.run([mainQN.predict, mainQN.rnn_state],
                                         feed_dict={mainQN.scalarInput: [s], mainQN.trainLength: 1, mainQN.state_in: state, mainQN.batch_size: 1, mainQN.prev_rewards: [[r]], mainQN.prev_actions: [a]})
                    a = a[0]

                s1P, r, d, obs = env.step(a)
                s1 = processState(s1P)
                total_steps += 1
                if(obs==True):
                    r=-2
                episodeBuffer.append(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))

                rAll += r
                s = s1
                sP = s1P
                state = state1

                if total_steps > pre_train_steps:
                    if e > endE:
                        e -= stepDrop

                    if total_steps % (update_freq) == 0:
                        updateTarget(targetOps, sess)

                        # Reset the recurrent layer's hidden state
                        state_train = (np.zeros([batch_size, hidden_size]), np.zeros(
                            [batch_size, hidden_size]))
                        # Get a random batch of experiences.
                        trainBatch, trace_length = myBuffer.sample(batch_size)

                        Q1 = sess.run(mainQN.predict,
                                      feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3]), mainQN.trainLength: trace_length, mainQN.state_in: state_train, mainQN.batch_size: batch_size,
                                                 mainQN.prev_rewards: np.vstack(trainBatch[:, 2]), mainQN.prev_actions: trainBatch[:, 1]})

                        Q2 = sess.run(targetQN.Qout,
                                      feed_dict={targetQN.scalarInput: np.vstack(trainBatch[:, 3]), targetQN.trainLength: trace_length, targetQN.state_in: state_train, targetQN.batch_size: batch_size,
                                                 targetQN.prev_rewards: np.vstack(trainBatch[:, 2]), targetQN.prev_actions: trainBatch[:, 1]})

                        end_multiplier = -(trainBatch[:, 4] - 1)
                        doubleQ = Q2[range(batch_size * trace_length), Q1]
                        targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                        # Update the network with our target values.

                        tt=sess.run(mainQN.updateModel,
                                     feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]), mainQN.targetQ: targetQ, mainQN.actions: trainBatch[:, 1], mainQN.trainLength: trace_length, mainQN.state_in: state_train, mainQN.batch_size: batch_size,
                                                mainQN.prev_rewards: np.vstack(np.insert(trainBatch[:, 2][1:], 0, 0)), mainQN.prev_actions: np.insert(trainBatch[:, 1][1:], 0, 0)})


                if d == True:
                    break

            # Add the episode to the experience buffer
            bufferArray = np.array(episodeBuffer)

            episodeBuffer = list(zip(bufferArray))
            myBuffer.add(episodeBuffer)
            jList.append(j)
            rList.append(rAll)

            if i % 1000 == 0:


                mean_reward = np.mean(rList[-1000:])
                summary = tf.Summary()
                summary.value.add(tag='Perf/Reward',
                                  simple_value=float(mean_reward))
                summary_writer.add_summary(summary, i)
                summary_writer.flush()
                print(i,e,mean_reward)

            if i>7000 and i%2000==0:
                saver.save(sess, path+str(int(mean_reward))+'_'+str(i)+'.cptk')
                if(mean_reward<20 and i>15000 and e_increase==False):
                    e=0.7
                    e_increase=True




# In[ ]:

def dqnrnn_eval(env_array):

    env = env_array[0]
    n_obs = env.n_obs
    n_actions = env.n_actions
    max_epLength = n_actions
    num_envs = len(env_array)


    tf.reset_default_graph()
    # We define the cells for the primary and target q-networks
    cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=hidden_size, state_is_tuple=True)
    cellT = tf.contrib.rnn.BasicLSTMCell(
        num_units=hidden_size, state_is_tuple=True)
    mainQN = Qnetwork(h_size, cell, 'main', n_obs, n_actions, hidden_size)
    targetQN = Qnetwork(h_size, cellT, 'target', n_obs, n_actions, hidden_size)
    saver = tf.train.Saver()


    test_epLength = n_actions

    df = pd.DataFrame(columns=['i', 'return', 'actions', 'ground_truth'])
    resum = 0

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('./models/temp/')

        best_reward = -1000
        models = ckpt.all_model_checkpoint_paths
        for i in range(len(models)):
            str = models[i]
            re = int(str.split('/')[-1].split('_')[0])
            if(re>=best_reward):
                model_path = str

        saver.restore(sess,model_path)

        for i in range(num_envs):
            env = env_array[i]
            sP = env.reset()
            s = processState(sP)
            actions = []
            d = False
            r=0
            a=0
            rAll = 0
            j = 0
            state = (np.zeros([1, hidden_size]), np.zeros([1, hidden_size]))
            while j < test_epLength:
                j += 1
                a, state1 = sess.run([mainQN.predict, mainQN.rnn_state],
                                     feed_dict={mainQN.scalarInput: [s], mainQN.trainLength: 1, mainQN.state_in: state, mainQN.batch_size: 1,
                                      mainQN.prev_rewards: [[r]], mainQN.prev_actions: [a]})
                a = a[0]

                s1P, r, d, obs = env.step(a)
                s1 = processState(s1P)

                if(obs==True):
                    s = s1
                    continue
                rAll += r
                actions.append(a)
                s = s1
                sP = s1P
                state = state1

                if d == True:
                    break

            if(d == False):
                a = 0

                actions.append(a)
                s1, r, d, obs = env.step(a)
                rAll += r

            df.loc[i] = [i, rAll, actions, env.ground_truth]
            #print(i, rAll, actions)
            resum += rAll

    print('----------------------------',resum /num_envs)
    return resum/num_envs



# In[ ]:
total_reward = np.zeros((5,5))
train_envs=[]
train_envs.append(get_train_envs(get_obs_theta(1,2)))
train_envs.append(get_train_envs(get_obs_theta(2,2)))
train_envs.append(get_train_envs(get_obs_theta(3,2)))
train_envs.append(get_train_envs(get_obs_theta(4,2)))
t=0
for i in train_envs:
    train_dqnrnn(i)
    for j in range(5):
        print(t,'-',j)
        r=dqnrnn_eval(get_eval_envs(j))
        total_reward[t][j]=r
    t=t+1
print(total_reward)
# In[ ]:
print(total_reward)

# In[ ]:
