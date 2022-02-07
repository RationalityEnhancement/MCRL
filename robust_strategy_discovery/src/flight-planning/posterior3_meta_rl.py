"""
code modified from https://github.com/awjuliani/Meta-RL
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
from helper1 import *

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

# my envs: ((1, 5, 9), (2, 6, 10), ((3, 4), (7, 8), (11, 12)))
# init: [None, Cat, Cat, Cat, Cat, Cat, Cat, Cat, Cat, Cat, Cat, Cat, Cat]
# tree: [[1, 5, 9], [2], [3, 4], [], [], [6], [7, 8], [], [], [10], [11, 12], [], []]

# Constants
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


# Classes
class AC_Network:
    def __init__(self, a_size, scope, trainer, n_obs):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(
                shape=[None, n_obs * 2], dtype=tf.float32)
            self.imageIn = tf.reshape(self.state, shape=[-1, 1, n_obs, 2])

            self.conv1 = slim.conv2d(
                inputs=self.imageIn, num_outputs=32, kernel_size=[1, 4], stride=[1, 2], padding='VALID', biases_initializer=None, scope=scope + '_conv1')

            self.conv2 = slim.conv2d(
                inputs=self.conv1, num_outputs=128, kernel_size=[1, 3], stride=[1, 2], padding='VALID', biases_initializer=None, scope=scope + '_conv2')

            self.conv3 = slim.conv2d(
                inputs=self.conv2, num_outputs=264, kernel_size=[1, 2], stride=[1, 2], padding='VALID', biases_initializer=None, scope=scope + '_conv3')

            self.fc = slim.fully_connected(slim.flatten(self.conv3), 64, activation_fn=tf.nn.elu)

            self.prev_rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.timestep = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            self.prev_actions_onehot = tf.one_hot(self.prev_actions, a_size+1, dtype=tf.float32)

            hidden = tf.concat([slim.flatten(self.fc), self.prev_rewards, self.prev_actions_onehot, self.timestep], 1)
            #hidden = tf.concat([slim.flatten(self.fc)],1)

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(400, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.prev_rewards)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size, time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 400])

            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(
                self.actions, a_size, dtype=tf.float32)

            # Output layers for policy and value estimations
            self.policy = slim.fully_connected(rnn_out, a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(
                                                   0.01),
                                               biases_initializer=None)
            self.value = slim.fully_connected(rnn_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(
                                                  1.0),
                                              biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(
                    shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(
                    self.policy * self.actions_onehot, [1])

                # Loss functions
                self.value_loss = 0.5 * \
                    tf.reduce_sum(tf.square(self.target_v -
                                            tf.reshape(self.value, [-1])))
                self.entropy = - \
                    tf.reduce_sum(self.policy * tf.log(self.policy + 1e-7))
                self.policy_loss = - \
                    tf.reduce_sum(
                        tf.log(self.responsible_outputs + 1e-7) * self.advantages)
                self.loss = .5 * self.value_loss + self.policy_loss - self.entropy * .05

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(
                    self.gradients, 5)

                # Apply local gradients to global network
                global_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(
                    zip(grads, global_vars))


class Worker:
    def __init__(self, game, name, a_size, trainer, model_path, global_episodes, n_obs):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(f'{MODEL_PATH}/tfb')

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(a_size, self.name, trainer, n_obs)
        self.update_local_ops = update_target_graph('global', self.name)
        self.env = game

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        states = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        timesteps = rollout[:, 3]
        prev_rewards = [0] + rewards[:-1].tolist()
        prev_actions = [0] + actions[:-1].tolist()
        values = rollout[:, 5]

        self.pr = prev_rewards
        self.pa = prev_actions
        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * \
            self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)


        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.state: np.stack(states, axis=0),
                     self.local_AC.prev_rewards: np.vstack(prev_rewards),
                     self.local_AC.prev_actions: prev_actions,
                     self.local_AC.actions: actions,
                     self.local_AC.timestep: np.vstack(timesteps),
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: rnn_state[0],
                     self.local_AC.state_in[1]: rnn_state[1]}
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads],
                                              feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, gamma, sess, coord, saver, train):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("start worker:", self.name)

        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                r = 0
                a = n_actions
                t = 0

                env = self.env[np.random.randint(0, NUM_SAMPLES)]
                sP = env._reset()
                s = process_state(sP)
                rnn_state = self.local_AC.state_init

                while episode_step_count < n_actions:
                    # Take an action using probabilities from policy network output.
                    a_dist, v, rnn_state_new = sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                                                        feed_dict={
                        self.local_AC.state: [s],
                        self.local_AC.prev_rewards: [[r]],
                        self.local_AC.timestep: [[t]],
                        self.local_AC.prev_actions: [a],
                        self.local_AC.state_in[0]: rnn_state[0],
                        self.local_AC.state_in[1]: rnn_state[1]})
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    rnn_state = rnn_state_new

                    s1P, r, d, obs = env._step(a)
                    s1 = process_state(s1P)

                    episode_buffer.append([s, a, r, t, d, v[0, 0]])
                    episode_values.append(v[0, 0])
                    episode_reward += r
                    total_steps += 1
                    t += 1
                    episode_step_count += 1
                    s = s1

                    if d:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                print('=====')
                print('Episode step count:', episode_step_count)
                print('Episode reward:', episode_reward)
                print('Episode values:', episode_values)
                print('Mean episode values:', np.mean(episode_values))
                print()

                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0 and train:
                    v_l, p_l, e_l, g_n, v_n = self.train(
                        episode_buffer, sess, gamma, 0.0)

                if SAVE_MODEL:
                    if episode_count % SAVE_FREQ == 0 and episode_count != 0:
                        if self.name == 'worker_0':
                            mean_reward = np.mean(self.episode_rewards[-SUMMARY_FREQ:])
                            saver.save(sess, f"{MODEL_PATH}/" + str(int(mean_reward))+
                                '_'+ str(episode_count) + '.cptk')
                        
                if episode_count % SUMMARY_FREQ == 0 and episode_count != 0:
                    if self.name == 'worker_0':
                        mean_reward = np.mean(self.episode_rewards[-SUMMARY_FREQ:])
                        mean_length = np.mean(self.episode_lengths[-SUMMARY_FREQ:])
                        mean_value = np.mean(self.episode_mean_values[-SUMMARY_FREQ:])
                        summary = tf.Summary()
                        summary.value.add(tag='Perf/Reward',
                                          simple_value=float(mean_reward))
                        if train:

                            summary.value.add(
                                tag='Losses/Value Loss', simple_value=float(v_l))
                            summary.value.add(
                                tag='Losses/Policy Loss', simple_value=float(p_l))
                            summary.value.add(tag='Losses/Entropy',
                                              simple_value=float(e_l))

                        self.summary_writer.add_summary(summary, episode_count)

                        self.summary_writer.flush()

                if self.name == 'worker_0':
                    sess.run(self.increment)

                    if episode_count >= NUM_EPISODES:

                        coord.request_stop()

                episode_count += 1


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
    # Read and normlize distribution values due to numerical precision problems
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
                                       sample_term_reward=False, env_type='old',
                                       repeat_cost=REPEAT_COST,
                                       repl_init=theta_might_be_init)]
    
    return posterior_envs


def make_envs_no_pos(theta_idx,cost=1, num_samples=1):
    envs = []
    theta = THETAS[theta_idx]
    for i in range(num_samples):
        init = convert_env(list(chain(*theta)))

        # Create a Mouselab environment
        envs += \
            [MouselabEnv.new_symmetric(BRANCHING, reward, seed=i, cost=cost, env_type='old',
                                       repl_init=init)]
    
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

    # CONSTANTS
    COST = 1
    GAMMA = .9  # discount rate for advantage estimation and reward discounting
    NUM_SAMPLES = 1000
    REPEAT_COST = 2
    
    NUM_EPISODES = 100000
    SAVE_FREQ = 10000
    SUMMARY_FREQ = 1000

    NUM_WORKERS = 4
    
    if train_flag:
        SAVE_MODEL = True
        LOAD_MODEL = False
        TRAIN = True
    else:
        SAVE_MODEL = False
        LOAD_MODEL = True
        TRAIN = False

    if train_flag:
        # Read posterior CSV file
        CSV_PATH = 'posterior3_old/two_observation/200x_samples_100_epochs/posteriors.csv'
        POSTERIOR_DF = pd.read_csv(CSV_PATH, index_col='theta_hat_idx')
        if posterior_flag:
            MODEL_PATH = f"models/posterior3/meta_rl/posterior/{theta_hat_idx}"
        else:
            MODEL_PATH = f"models/posterior3/meta_rl/no_posterior/{theta_hat_idx}"
        
        try:
            os.makedirs(MODEL_PATH)
        except Exception as e:
            print(str(e))

        if posterior_flag:
            # Read posterior distribution for current theta_hat
            posterior_dist = POSTERIOR_DF.loc[theta_hat_idx]
            envs = make_envs(posterior_dist, num_samples=NUM_SAMPLES)
        else:
            envs = make_envs_no_pos(theta_hat_idx, num_samples=NUM_SAMPLES)
            
        # Mostly unchanged...
        env = envs[np.random.randint(0, NUM_SAMPLES)]
        n_obs = env.n_obs
        a_size = max_epLength = n_actions = env.n_actions
        # Create workers & train
        with tf.device("/cpu:0"):
            global_episodes = tf.Variable(
                0, dtype=tf.int32, name='global_episodes', trainable=False)
            trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
            
            # Generate global network
            master_network = AC_Network(a_size, 'global', None, n_obs)

            # Set workers to number of available CPU threads
            # num_workers = multiprocessing.cpu_count()
            
            # Create worker classes
            workers = []
            for i in range(NUM_WORKERS):
                workers.append(Worker(envs, i, a_size, trainer,
                                      MODEL_PATH, global_episodes, n_obs))
            saver = tf.train.Saver(max_to_keep=10000)  # max_to_keep=5
        
        #
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            if LOAD_MODEL:
                print('Loading Model...')
                ckpt = tf.train.get_checkpoint_state(f"{MODEL_PATH}")
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())
        
            worker_threads = []
            for worker in workers:
                def worker_work():
                    return worker.work(GAMMA, sess, coord, saver, TRAIN)
                
                thread = threading.Thread(target=(worker_work))
                thread.start()
                worker_threads.append(thread)
                
            coord.join(worker_threads)

    else:
        if final_avg:
            freq_df = pd.read_csv('posterior3_old/theta_hat_freq.csv',
                                  index_col='theta_hat_idx')
            freq = freq_df.values
            prob = freq / np.sum(freq)
            total_reward = 0
            if posterior_flag:
                PATH = f"models/posterior3/meta_rl/posterior/eval/"
            else:
                PATH = f"models/posterior3/meta_rl/no_posterior/eval/"

            for theta_hat_idx in range(len(THETAS)):
                CSV_PATH = PATH + str(theta_hat_idx) + '.csv'
                tt = pd.read_csv(CSV_PATH, sep=',', index_col=0)
                r = np.array(tt['return'])
                reward = r.mean()
                total_reward += prob[theta_hat_idx] * reward
            print(f"Posterior = {posterior_flag} Total Reward = {total_reward}")

            try:
                # Create target Directory
                os.makedirs(PATH)
                print("Directory ", PATH, "created.")
            except FileExistsError:
                print("Directory ", PATH, "already exists.")
            PATH1 = PATH + 'Results'
            
            # Write results
            np.save(PATH1, total_reward)

        else:
            # Generate model path
            if posterior_flag:
                MODEL_PATH = f"models/posterior3/meta_rl/posterior/{theta_hat_idx}"
                SAVE_PATH = f"models/posterior3/meta_rl/posterior/eval/"
            elif posterior_flag:
                MODEL_PATH = f"models/posterior3/meta_rl/no_posterior/{theta_hat_idx}"
                SAVE_PATH = f"models/posterior3/meta_rl/no_posterior/eval/"
        
            try:
                os.makedirs(MODEL_PATH)
            except Exception as e:
                print(str(e))
            CSV_PATH = 'posterior3_old/two_observation/200x_samples_100_epochs/posteriors.csv'
            POSTERIOR_DF = pd.read_csv(CSV_PATH, index_col='theta_hat_idx')
            
            # Read posterior distribution for current theta_hat
            posterior_dist = POSTERIOR_DF.loc[theta_hat_idx]
            posterior_envs = make_envs(posterior_dist, num_samples=NUM_SAMPLES)

            # Mostly unchanged...
            env = posterior_envs[np.random.randint(0, NUM_SAMPLES)]
            n_obs = env.n_obs
            a_size = max_epLength = n_actions = env.n_actions
            
            tf.reset_default_graph()
            
            # Create workers & train
            with tf.device("/cpu:0"):
                # Generate global network
                master_network = AC_Network(a_size, 'global', None, n_obs)

                # Set workers to number of available CPU threads
                # num_workers = multiprocessing.cpu_count()

                # Create worker classes
                # Change: Commented next 4 lines
                # workers = []
                # for i in range(NUM_WORKERS):
                #     workers.append(Worker(posterior_envs, i, a_size, trainer,
                #                           MODEL_PATH, global_episodes, n_obs))
                saver = tf.train.Saver(max_to_keep=10000)  # max_to_keep=5


            # with tf.Session() as sess:
            #     coord = tf.train.Coordinator()
            #     if LOAD_MODEL:
            #         print('Loading Model...')
            #         ckpt = tf.train.get_checkpoint_state(f"{MODEL_PATH}")
            #         saver.restore(sess, ckpt.model_checkpoint_path)
            #     else:
            #         sess.run(tf.global_variables_initializer())
            #
            #     worker_threads = []
            #     for worker in workers:
            #         def worker_work():
            #             return worker.work(GAMMA, sess, coord, saver, TRAIN)
            #
            #         thread = threading.Thread(target=(worker_work))
            #         thread.start()
            #         worker_threads.append(thread)
            #
            #     coord.join(worker_threads)


            # ===== Evaluation =====
            eval_samples_df = pd.read_csv('posterior/eval_samples.csv')
            thetas = eval_samples_df.iloc[theta_hat_idx][1:]
            
            posterior_envs = []

            df = pd.DataFrame(
                columns=['i', 'return', 'env_type', 'actions', 'ground_truth'])
            with tf.Session() as sess:
                print('Loading Model...')
                # print(MODEL_PATH)
                dirs = os.listdir(MODEL_PATH)
                score = []
                iteration = []
                cptk_names = []
                for file in dirs:
                    t = file.split("_")
                    if len(t) <= 1: # Should take checkpoint, tfb, and iteration only files into account
                        continue
                    c = file.split(".")
                    it = t[1].split(".")
                    iteration.append(int(it[0]))
                    cptk_names.append(c[0])

                    score.append(int(t[0]))
                    
                m = max(score)
                MAX_INDEX = [i for i, j in enumerate(score) if j == m]
                itt = []
                for _, i in enumerate(MAX_INDEX):
                    itt.append(iteration[i])
                    
                MAX_INDEX = iteration.index(max(itt))
                MODEL_FILE = MODEL_PATH + "/" + cptk_names[MAX_INDEX] + ".cptk.meta"
                print("Restoring: {} Max Score = {}".format(MODEL_FILE, m))
                saver = tf.train.import_meta_graph(MODEL_FILE)
                saver.restore(sess, MODEL_FILE[:-5])
                
                resum = 0
                
                for i, theta_idx in enumerate(thetas):
                    # Sample from the posterior
                    theta = THETAS[theta_idx]
            
                    # Convert theta_might_be to MouselabEnv init vector
                    theta_might_be_init = convert_env(list(chain(*theta)))
            
                    # Create a Mouselab environment
                    posterior_envs += \
                        [MouselabEnv.new_symmetric(BRANCHING, reward, seed=i, cost=COST,
                                                   sample_term_reward=False,
                                                   env_type='old',
                                                   repeat_cost=REPEAT_COST,
                                                   repl_init=theta_might_be_init)]
            
                    episode_reward = 0
                    episode_step_count = 0
                    d = False
                    r = 0
                    a = n_actions
                    t = 0
                
                    actions = []
                    env = posterior_envs[i]
                    sP = env._reset()
                    s = process_state(sP)
                    rnn_state = master_network.state_init
                
                    while episode_step_count < max_epLength:
                        # Take an action using probabilities from policy network output.
                        a_dist, v, rnn_state_new = sess.run(
                            [master_network.policy, master_network.value,
                             master_network.state_out],
                            feed_dict={
                                master_network.state:        [s],
                                master_network.prev_rewards: [[r]],
                                master_network.timestep:     [[t]],
                                master_network.prev_actions: [a],
                                master_network.state_in[0]:  rnn_state[0],
                                master_network.state_in[1]:  rnn_state[1]})
                        a = np.random.choice(a_dist[0], p=a_dist[0])
                        a = np.argmax(a_dist == a)
                    
                        rnn_state = rnn_state_new
                    
                        s1P, r, d, obs = env._step(a)
                        s1 = process_state(s1P)
                    
                        if obs:
                            s = s1
                            t += 1
                            episode_step_count += 1
                            continue
                    
                        episode_reward += r
                        actions.append(a)
                        episode_step_count += 1
                        s = s1
                        t += 1
                    
                        if d:
                            break
                
                    if d:
                        a = 0
                        actions.append(a)
                        s1, r, d, obs = env._step(a)
                        episode_reward += r
                
                
                    df.loc[i] = [i, episode_reward, theta_idx, actions,
                                 env.ground_truth]
                    resum += episode_reward

            print(resum / len(thetas))
           
            try:
                os.makedirs(SAVE_PATH)
            except Exception as e:
                print(str(e))

            df.to_csv(f'{SAVE_PATH}/{theta_hat_idx}.csv')
