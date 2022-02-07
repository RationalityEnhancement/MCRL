'''
code modified from https://github.com/awjuliani/DeepRL-Agents
'''

# In[ ]:

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
from distributions import smax, cmax, sample
from posterior import get_train_envs,get_particular_theta


# In[ ]:

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

    mask = mask[:] * 60


    state_arr = [
    [state[3],state[4],state[5]],
    [state[2],state[6],state[10]],
    [state[1],state[14],state[27]],
    ]

    mask_arr = [
    [mask[3],mask[4],mask[5]],
    [mask[2],mask[6],mask[10]],
    [mask[1],mask[14],mask[27]],
    ]

    out = np.stack((state_arr,mask_arr), axis=0)
    return out

# In[ ]:

def performAction(env,action):
    if action==0:
        return env.step(action)
    else:
        if action==1:
            ac_set=[1]

        if action==2:
            ac_set=[14]

        if action==3:
            ac_set=[27]

        if action==4:
            ac_set=[2,15,28]

        if action==5:
            ac_set =[6,19,32]

        if action==6:
            ac_set=[10,23,36]

        if action==7:
            ac_set=[3,7,11,16,20,24,29,33,37]

        if action==8:
            ac_set=[4,8,12,17,21,25,30,34,38]

        if action==9:
            ac_set=[5,9,13,18,22,26,31,35,39]


        for i in ac_set:
            s1, r, d, obs = env.step(i)
        if(obs==True):
            r=env.repeat_cost
        else:
            r=env.cost

        return s1,r,d,obs

# In[ ]:


class Qnetwork():
    def __init__(self, h_size, rnn_cell, myScope,n_actions, hidden_size):

        self.imageIn = tf.placeholder(shape=[None,2,3,3], dtype=tf.float32)


        self.conv1 = slim.conv2d(inputs=self.imageIn, num_outputs=32, kernel_size=[1, 1], stride=[
                                 1, 1], padding='VALID', data_format='NCHW',biases_initializer=None)

        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=64, kernel_size=[1, 1], stride=[
                                 1, 1], padding='VALID', data_format='NCHW',biases_initializer=None)

        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=128, kernel_size=[1, 3], stride=[
                                 1, 1], padding='VALID', data_format='NCHW',biases_initializer=None)

        self.conv3 = slim.conv2d(inputs=self.conv3, num_outputs=h_size, kernel_size=[3, 1], stride=[
                                1, 1], padding='VALID', data_format='NCHW',biases_initializer=None)


        self.prev_rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.prev_actions_onehot = tf.one_hot(self.prev_actions, n_actions, dtype=tf.float32)

        self.fc1 = slim.fully_connected(tf.concat([self.prev_actions_onehot,self.prev_rewards], 1),32,activation_fn=tf.nn.elu)
        self.hidden1 = tf.concat([slim.flatten(self.conv3),self.fc1], 1)
        #self.hidden1 = slim.flatten(self.conv3)

        self.trainLength = tf.placeholder(dtype=tf.int32)
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])

        self.hidden=tf.reshape(self.hidden1, [self.batch_size, self.trainLength, hidden_size])


        self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn( inputs=self.hidden, cell=rnn_cell, dtype=tf.float32, initial_state=self.state_in, scope=myScope + '_rnn')
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

# In[ ]:

class experience_buffer():
    def __init__(self, buffer_size=3000):
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
update_freq = 2  # How often to perform a training step.
y = .99  # Discount factor on the target Q-values
startE = 1  # Starting chance of random action
endE = 0.15  # Final chance of random action
num_episodes = 20000  # How many steps of training to reduce startE to endE.
annealing_steps = 3000 * 10
pre_train_steps = 2000 * 10
path = "./models/temp/"  # The path to save our model to.
learning_rate = 1e-4
time_per_step = 1  # Length of each step used in gif creation
tau = 0.001
h_size = 200
hidden_size = h_size+32
# In[ ]:
def train_dqnrnn(env_array):

    n_actions = 10
    max_epLength = n_actions
    num_envs = len(env_array)

    tf.reset_default_graph()
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
    cellT = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
    mainQN = Qnetwork(h_size, cell, 'main', n_actions, hidden_size)
    targetQN = Qnetwork(h_size, cellT, 'target', n_actions, hidden_size)

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
            rAll = 0
            j = 0
            r=0
            a=10
            random_ac = random.sample(range(0,n_actions), n_actions)
            # Reset the recurrent layer's hidden state
            state = (np.zeros([1, hidden_size]), np.zeros([1, hidden_size]))
            # The Q-Network
            while j < max_epLength:
                j += 1
                # Choose an action by greedily (with e chance of random action) from the Q-network
                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    state1 = sess.run(mainQN.rnn_state,
                                      feed_dict={mainQN.imageIn: [s], mainQN.trainLength: 1, mainQN.state_in: state, mainQN.batch_size: 1, mainQN.prev_rewards: [[r]], mainQN.prev_actions: [a]})
                    a = random_ac[0]
                    random_ac = random_ac[1:]

                else:
                    a, state1 = sess.run([mainQN.predict, mainQN.rnn_state],
                                         feed_dict={mainQN.imageIn: [s], mainQN.trainLength: 1, mainQN.state_in: state, mainQN.batch_size: 1, mainQN.prev_rewards: [[r]], mainQN.prev_actions: [a]})
                    a = a[0]

                s1P, r, d, obs = performAction(env,a)
                s1 = processState(s1P)
                total_steps += 1
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

                        f=trainBatch[:,3].tolist()
                        m=np.empty((0,2,3,3))
                        for g in f:
                            m=np.append(m,[g],axis=0)
                        in3 = m

                        f=trainBatch[:,0].tolist()
                        m=np.empty((0,2,3,3))
                        for g in f:
                            m=np.append(m,[g],axis=0)
                        in1 = m


                        Q1 = sess.run(mainQN.predict,
                                      feed_dict={mainQN.imageIn: in3, mainQN.trainLength: trace_length, mainQN.state_in: state_train, mainQN.batch_size: batch_size,
                                                 mainQN.prev_rewards: np.vstack(trainBatch[:, 2]), mainQN.prev_actions: trainBatch[:, 1]})

                        Q2 = sess.run(targetQN.Qout,
                                      feed_dict={targetQN.imageIn: in3, targetQN.trainLength: trace_length, targetQN.state_in: state_train, targetQN.batch_size: batch_size,
                                                 targetQN.prev_rewards: np.vstack(trainBatch[:, 2]), targetQN.prev_actions: trainBatch[:, 1]})

                        end_multiplier = -(trainBatch[:, 4] - 1)
                        doubleQ = Q2[range(batch_size * trace_length), Q1]
                        targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                        # Update the network with our target values.

                        tt=sess.run(mainQN.updateModel,
                                     feed_dict={mainQN.imageIn: in1, mainQN.targetQ: targetQ, mainQN.actions: trainBatch[:, 1], mainQN.trainLength: trace_length, mainQN.state_in: state_train, mainQN.batch_size: batch_size,
                                                mainQN.prev_rewards: np.vstack(np.insert(trainBatch[:, 2][1:], 0, 0)), mainQN.prev_actions: np.insert(trainBatch[:, 1][1:], 0, 0)})


                if d == True:
                    break

            # Add the episode to the experience buffer
            bufferArray = np.array(episodeBuffer)

            episodeBuffer = list(zip(bufferArray))
            myBuffer.add(episodeBuffer)
            jList.append(j)
            rList.append(rAll)

            if (i+1) % 1000 == 0:


                mean_reward = np.mean(rList[-1000:])
                summary = tf.Summary()
                summary.value.add(tag='Perf/Reward',
                                  simple_value=float(mean_reward))
                summary_writer.add_summary(summary, i)
                summary_writer.flush()
                print(i,e,mean_reward)

            if i>13000 and i%2000==0:
                saver.save(sess, path+str(int(mean_reward))+'_'+str(i)+'.cptk')

# In[ ]:

#env_array = get_particular_theta()
env_array,pos = get_train_envs(1)
train_dqnrnn(env_array)

# In[ ]:


def dqnrnn_eval(env_array):

    n_actions = 10
    max_epLength = n_actions
    num_envs = len(env_array)


    tf.reset_default_graph()
    # We define the cells for the primary and target q-networks
    cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=hidden_size, state_is_tuple=True)
    cellT = tf.contrib.rnn.BasicLSTMCell(
        num_units=hidden_size, state_is_tuple=True)
    mainQN = Qnetwork(h_size, cell, 'main',  n_actions, hidden_size)
    targetQN = Qnetwork(h_size, cellT, 'target',  n_actions, hidden_size)
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
                best_reward=re

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
                                     feed_dict={mainQN.imageIn: [s], mainQN.trainLength: 1, mainQN.state_in: state, mainQN.batch_size: 1,
                                      mainQN.prev_rewards: [[r]], mainQN.prev_actions: [a]})
                a = a[0]

                s1P, r, d, obs = performAction(env,a)
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
            print(i, rAll, actions)
            resum += rAll

    print('----------------------------',resum /num_envs)
    df.to_csv('csv/temp1.csv')
    return resum/num_envs




# In[ ]:
env_array,pos = get_train_envs(1)
dqnrnn_eval(env_array)

# In[ ]:
