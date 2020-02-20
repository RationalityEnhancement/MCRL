"""
code modified from https://github.com/awjuliani/DeepRL-Agents
"""

import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import scipy.signal
from helper import *

from random import choice
from time import sleep
from time import time
import glob
import shutil
from distributions import Categorical, Normal
import pandas as pd

import argparse
from itertools import chain
from mouselab import MouselabEnv
from posterior3 import THETAS, OUTCOMES, get_theta, \
    get_two_observation_theta_hat, get_confused_theta_hat, get_random_theta_hat
from tqdm import tqdm
import pickle

BRANCHING = [3, 1, 2]

MDP_TO_THETA = {
    # Level 1
    1: 0,
    5: 1,
    9: 2,
    
    # Level 2
    2: 3,
    6: 4,
    10: 5,
    
    # Level 3
    # Branch 1
    3: 6,
    4: 6,
    
    # Branch 2
    7: 7,
    8: 7,
    
    # Branch 3
    11: 8,
    12: 8
}


def convert_env(theta):
    init = [None]
    
    for idx in range(min(MDP_TO_THETA), max(MDP_TO_THETA)+1):
        theta_idx = MDP_TO_THETA[idx]
        node_type = theta[theta_idx]
        init += [Categorical(OUTCOMES[node_type])]

    return init


def reward(depth):
    if depth == 1:
        return Categorical(OUTCOMES['L'] + OUTCOMES['M'])
    if depth == 2:
        return Categorical(OUTCOMES['L'] + OUTCOMES['M'])
    if depth == 3:
        return Categorical(OUTCOMES['M'] + OUTCOMES['H'])


def make_envs(posterior_dist, cost=1, num_samples=1):
    # Read and normalize distribution values due to numerical precision problems
    dist = posterior_dist.values
    dist = dist / np.sum(dist)

    posterior_envs = []

    for i in range(num_samples):
        # Sample from the posterior
        theta_might_be_idx = np.random.choice(np.arange(len(THETAS)), p=dist)
        theta_might_be = THETAS[theta_might_be_idx]
        
        # Convert theta_might_be to MouselabEnv init vector
        theta_might_be_init = convert_env(list(chain(*theta_might_be)))

        # Create a Mouselab environment
        posterior_envs += \
            [MouselabEnv.new_symmetric(BRANCHING, reward, seed=i, cost=cost,
                                       env_type='old',
                                       repl_init=theta_might_be_init)]
    
    return posterior_envs


def make_envs_no_pos(theta_idx, cost=1, num_samples=1):
    envs = []
    theta = THETAS[theta_idx]
    for i in range(num_samples):
        init = convert_env(list(chain(*theta)))
        
        # Create a Mouselab environment
        envs += \
            [MouselabEnv.new_symmetric(BRANCHING, reward, seed=i, cost=cost,
                                       env_type='old', repl_init=init)]
    
    return envs


def make_true_envs_randomly(seed=0, cost=1, num=1000):
    envs = []
    np.random.seed(seed)
    for i in range(num):
        idx = np.random.choice(np.arange(len(THETAS)))
        theta = THETAS[idx]
        
        # Convert theta_might_be to MouselabEnv init vector
        init = convert_env(list(chain(*theta)))

        # Create a Mouselab environment
        envs += \
            [MouselabEnv.new_symmetric(BRANCHING, reward, seed=i, cost=cost,
                                       env_type='old', repl_init=init)]
    
    return envs


def process_state(instate):
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


class Qnetwork():
    def __init__(self, h_size, rnn_cell, myScope, n_obs, n_actions, hidden_size):

        self.scalarInput = tf.placeholder(
            shape=[None, n_obs * 2], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 1, n_obs, 2])

        self.conv1 = slim.conv2d(inputs=self.imageIn, num_outputs=32,
                                 kernel_size=[1, 4], stride=[1, 2],
                                 padding='VALID', biases_initializer=None,
                                 scope=myScope + '_conv1')

        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=90,
                                 kernel_size=[1, 3], stride=[1, 2],
                                 padding='VALID', biases_initializer=None,
                                 scope=myScope + '_conv2')

        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=h_size,
                                 kernel_size=[1, 2], stride=[1, 2],
                                 padding='VALID', biases_initializer=None,
                                 scope=myScope + '_conv3')


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


class experience_buffer:
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
    saver = tf.train.Saver(max_to_keep=10000)

    # Set the rate of random action decrease.
    e = startE
    stepDrop = (startE - endE) / annealing_steps

    # create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    total_steps = 0
    e_increase=True

    # try:
    #     shutil.rmtree('./tfb')
    # except OSError:
    #     print ("no dir")
    #
    # files = glob.glob(MODEL_PATH + '/ *')
    # for f in files:
    #     os.remove(f)

    summary_writer = tf.summary.FileWriter(f"{MODEL_PATH}/tfb")

    with tf.Session() as sess:

        sess.run(init)

        # Set the target network to be equal to the primary network.
        updateTarget(targetOps, sess)
        for i in range(NUM_EPISODES):
            episodeBuffer = []
            # Reset environment and get first new observation
            env = env_array[np.random.randint(0, num_envs)]
            sP = env._reset()
            s = process_state(sP)
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

                s1P, r, d, obs = env._step(a)
                s1 = process_state(s1P)
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
                        targetQ = trainBatch[:, 2] + (GAMMA * doubleQ * end_multiplier)
                        # Update the network with our target values.

                        tt=sess.run(mainQN.updateModel,
                                     feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]), mainQN.targetQ: targetQ, mainQN.actions: trainBatch[:, 1], mainQN.trainLength: trace_length, mainQN.state_in: state_train, mainQN.batch_size: batch_size,
                                                mainQN.prev_rewards: np.vstack(np.insert(trainBatch[:, 2][1:], 0, 0)), mainQN.prev_actions: np.insert(trainBatch[:, 1][1:], 0, 0)})


                if d:
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
                print(i, e, mean_reward)

            if i > 7000 and i % 2000 == 0:
                saver.save(sess, f"{MODEL_PATH}/" + str(int(mean_reward))+
                           '_'+str(i)+'.cptk')
                if mean_reward<20 and i>15000 and e_increase==False:
                    e = 0.7
                    e_increase = True


def eval(theta_hat_idx, cost=1, repeat_cost=2, num_samples=1000):
    eval_samples_df = pd.read_csv('posterior/eval_samples.csv')

    thetas = eval_samples_df.iloc[theta_hat_idx][1:]

    envs = []
    for i, theta_idx in enumerate(thetas):
        theta = THETAS[theta_idx]
        init = convert_env(list(chain(*theta)))
        envs += [MouselabEnv.new_symmetric(BRANCHING, reward, seed=i, cost=cost,
                                           env_type='old', repl_init=init)]
        
    if posterior_flag:
        pathname  = f"dqnrnn/posterior/models/{theta_hat_idx}"
    else:
        pathname  = f"dqnrnn/no_posterior/models/{theta_hat_idx}"

    return dqnrnn_eval(envs, theta_hat_idx, pathname)


def avg_eval(path):
    # Calculate weighted average over all theta_hats
    freq_df = pd.read_csv('./posterior/theta_hat_freq.csv',
                          index_col='theta_hat_idx')
    freq = freq_df.values
    prob = freq / np.sum(freq)
    total_reward = 0
    for theta_hat_idx in range(len(THETAS)):
        reward = np.load(path + str(theta_hat_idx) + '.npy')
        total_reward += prob[theta_hat_idx] * reward
    return total_reward


def dqnrnn_eval(env_array, theta_hat_idx, pathname):
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
    saver = tf.train.Saver(max_to_keep=10000)


    test_epLength = n_actions

    df = pd.DataFrame(columns=['i', 'return', 'actions', 'ground_truth'])
    resum = 0

    with tf.Session() as sess:
        print("Pathname = {}".format(pathname))
        pathname = os.path.normpath(pathname)
        print("Pathname = {}".format(pathname))
        dirs = os.listdir(pathname)
        score = []
        iteration = []
        cptk_names = []
        for file in dirs:
            t = file.split("_")
            if(len(t)<=1): # Should take checkpoint, tfb, and iteration only files into account
                continue
            c = file.split(".")
            it = t[1].split(".")
            iteration.append(int(it[0]))
            cptk_names.append(c[0])
            score.append(int(t[0]))
        m = max(score)
        MAX_INDEX = [i for i, j in enumerate(score) if j == m]
        itt = []
        for _,i in enumerate(MAX_INDEX):
            itt.append(iteration[i])
            
        MAX_INDEX = iteration.index(max(itt))
        MODEL_FILE = pathname + "/" + cptk_names[MAX_INDEX] + ".cptk.meta"
        print("Restoring: {} Max Score = {}".format(MODEL_FILE , m))
        saver = tf.train.import_meta_graph(MODEL_FILE)
        saver.restore(sess, MODEL_FILE[:-5])

        for i in range(num_envs):
            env = env_array[i]
            sP = env._reset()
            s = process_state(sP)
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
                                     feed_dict={mainQN.scalarInput: [s],
                                                mainQN.trainLength: 1,
                                                mainQN.state_in: state,
                                                mainQN.batch_size: 1,
                                                mainQN.prev_rewards: [[r]],
                                                mainQN.prev_actions: [a]})
                a = a[0]

                s1P, r, d, obs = env._step(a)
                s1 = process_state(s1P)

                if obs:
                    s = s1
                    continue
                rAll += r
                actions.append(a)
                s = s1
                sP = s1P
                state = state1

                if d:
                    break

            if not d:
                a = 0

                actions.append(a)
                s1, r, d, obs = env._step(a)
                rAll += r

            df.loc[i] = [i, rAll, actions, env.ground_truth]
            #print(i, rAll, actions)
            resum += rAll

    print('----------------------------',resum /num_envs)
    return resum/num_envs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('theta_hat_idx', type=int,
                        help='Index of theta hat from the enumerated list')
    parser.add_argument('train_flag', type=int,
                        help='1 for training. 0 for Testing')
    parser.add_argument('posterior_flag', type=int,
                        help='1 for evaluating models trained on posterior')
    parser.add_argument('final_avg', type=int,
                        help='1 for final average')
    args = parser.parse_args()

    theta_hat_idx = args.theta_hat_idx
    posterior_flag = args.posterior_flag
    final_avg = args.final_avg
    train_flag = args.train_flag

    # print("theta_hat_idx = {}".format(theta_hat_idx))
    # print("train_flag = {}".format(train_flag))
    # print("posterior_flag = {}".format(posterior_flag))
    # print("final_avg = {}".format(final_avg))

    NUM_SAMPLES = 1000
    REPEAT_COST = 2
    NUM_EPISODES = 100000  # 20000
    SAVE_FREQ = 10000  # 3000
    SUMMARY_FREQ = 1000  # 500

    NUM_WORKERS = 4
    if train_flag:
        SAVE_MODEL = True
        LOAD_MODEL = False
        TRAIN = True

    else:
        SAVE_MODEL = False
        LOAD_MODEL = True
        TRAIN = False

    # Setting the training parameters
    batch_size = 1  # How many experience traces to use for each training step.
    update_freq = 5  # How often to perform a training step.
    GAMMA = .99  # Discount factor on the target Q-values
    startE = 1  # Starting chance of random action
    endE = 0.15  # Final chance of random action

    annealing_steps = 4000 * 10
    pre_train_steps = 2000 * 10
    learning_rate = 1e-4
    time_per_step = 1  # Length of each step used in gif creation
    summaryLength = 100  # Number of epidoes to periodically save for analysis
    tau = 0.001
    h_size = 300
    hidden_size = h_size

    if train_flag:
        # Read posterior CSV file
        CSV_PATH = 'posterior3/two_observation/200x_samples_100_epochs/posteriors.csv'
        POSTERIOR_DF = pd.read_csv(CSV_PATH, index_col='theta_hat_idx')

        # Read posterior distribution for current theta_hat
        if posterior_flag:
            MODEL_PATH = f"models/posterior3/dqnrnn/posterior/{theta_hat_idx}"
        else:
            MODEL_PATH = f"models/posterior3/dqnrnn/no_posterior/{theta_hat_idx}"
        
        try:
            os.makedirs(MODEL_PATH)
        except Exception as e:
            print(str(e))
        
        if posterior_flag:
            posterior_dist = POSTERIOR_DF.loc[theta_hat_idx]
            envs = make_envs(posterior_dist, num_samples=NUM_SAMPLES)
        else:
            envs = make_envs_no_pos(theta_hat_idx, num_samples=NUM_SAMPLES)
            
        # Mostly unchanged...
        env = envs[np.random.randint(0, NUM_SAMPLES)]
        n_obs = env.n_obs
        a_size = max_epLength = n_actions = env.n_actions

        tf.reset_default_graph()

        train_dqnrnn(envs)

    else:
        tf.reset_default_graph()
        if final_avg == 0:
            # print("Calling Eval")
            rewards = eval(theta_hat_idx,posterior_flag)

            if posterior_flag:
                PATH = f"models/posterior3/dqnrnn/posterior/eval/"
            else:
                PATH = f"models/posterior3/dqnrnn/no_posterior/eval/"
            try:
                os.makedirs(PATH)
            except Exception as e:
                print(str(e))

            PATH1 = PATH + str(theta_hat_idx)
            print("Writing Results of {} in {}".format(theta_hat_idx, PATH1))
            np.save(PATH1, rewards)

        if final_avg:
            if posterior_flag:
                PATH = f"models/posterior3/dqnrnn/posterior/eval/"
            else:
                PATH = f"models/posterior3/dqnrnn/no_posterior/eval/"
                
            try:
                os.makedirs(PATH)
            except Exception as e:
                print(str(e))

            rewards = avg_eval(PATH)

            if posterior_flag:
                PATH = f"models/posterior3/dqnrnn/posterior/final/eval/"
            else:
                PATH = f"models/posterior3/dqnrnn/no_posterior/final/eval/"
            try:
                os.makedirs(PATH)
            except Exception as e:
                print(str(e))
            PATH1 = PATH + 'Results'
            # print("Writing Final Results")
            print("Posterior = {} Total Reward = {}".format(posterior_flag, rewards))
            np.save(PATH1, rewards)
