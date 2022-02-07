
# In[ ]:


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
from mouselab import MouselabEnv
from distributions import Categorical, Normal
import pandas as pd
from generate_environments import *


# In[ ]:


def make_envs1(cost=1, n=100):
    repeat_cost = 2
    def reward(depth):
        if depth == 1:
            return Categorical([-48, -24, 24, 48])
        elif depth == 2:
            return Categorical([-8, -4, 4, 8])
        elif depth == 3:
            return Categorical([-4, -2, 2, 4])


    branching = [4, 1, 1]
    return [MouselabEnv.new_symmetric(branching, reward, seed=i, cost=cost, sample_term_reward=False, env_type='new', repeat_cost=repeat_cost)
            for i in range(n)]


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
    state = state[1:]
    mask = mask[1:] * 60
    out = np.stack((state, mask), axis=1).flatten()

    return out


# In[ ]:


class AC_Network():
    def __init__(self, a_size, scope, trainer, n_obs):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(
                shape=[None, n_obs * 2], dtype=tf.float32)
            self.imageIn = tf.reshape(self.state, shape=[-1, 1, n_obs, 2])

            self.conv1 = slim.conv2d(
                inputs=self.imageIn, num_outputs=32, kernel_size=[1, 4], stride=[1, 2], padding='VALID', biases_initializer=None, scope=scope + '_conv1')

            self.conv2 = slim.conv2d(
                inputs=self.conv1, num_outputs=64, kernel_size=[1, 3], stride=[1, 2], padding='VALID', biases_initializer=None, scope=scope + '_conv2')


            self.fc = slim.flatten(self.conv2)

            self.prev_rewards = tf.placeholder(
                shape=[None, 1], dtype=tf.float32)
            self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.timestep = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            self.prev_actions_onehot = tf.one_hot(
                self.prev_actions, a_size + 1, dtype=tf.float32)

            self.fc2 = tf.concat([self.prev_rewards, self.prev_actions_onehot], 1)

            hidden = tf.concat([self.fc, self.fc2], 1)

            # hidden = tf.concat([slim.flatten(self.fc)],1)

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(200, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)

            rnn_in = tf.expand_dims(hidden, [0])
            step_size= tf.shape(self.prev_rewards)[: 1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state = state_in, time_major = False)
            lstm_c, lstm_h=lstm_state
            self.state_out=(lstm_c[:1, :], lstm_h[:1, :])
            rnn_out=tf.reshape(lstm_outputs, [-1, 200])

            self.actions=tf.placeholder(shape = [None], dtype = tf.int32)
            self.actions_onehot=tf.one_hot(
                self.actions, a_size, dtype = tf.float32)

            # Output layers for policy and value estimations
            self.policy=slim.fully_connected(rnn_out, a_size,
                                               activation_fn = tf.nn.softmax,
                                               weights_initializer = normalized_columns_initializer(
                                                   0.01),
                                               biases_initializer = None)
            self.value=slim.fully_connected(
                rnn_out, 1, activation_fn = None, weights_initializer = normalized_columns_initializer(1.0), biases_initializer = None)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.target_v=tf.placeholder(
                    shape = [None], dtype = tf.float32)
                self.advantages=tf.placeholder(
                    shape = [None], dtype = tf.float32)

                self.responsible_outputs=tf.reduce_sum(
                    self.policy * self.actions_onehot, [1])

                # Loss functions
                self.value_loss=0.5 * \
                    tf.reduce_sum(tf.square(self.target_v - \
                                  tf.reshape(self.value, [-1])))
                self.entropy=- \
                    tf.reduce_sum(self.policy * tf.log(self.policy + 1e-7))
                self.policy_loss=- \
                    tf.reduce_sum(
                        tf.log(self.responsible_outputs + 1e-7) * self.advantages)

                self.loss = 0.1 * self.value_loss + self.policy_loss - self.entropy * .0005

                # Get gradients from local network using local losses
                local_vars=tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients=tf.gradients(self.loss, local_vars)
                self.var_norms=tf.global_norm(local_vars)
                grads, self.grad_norms=tf.clip_by_global_norm(
                    self.gradients, 9)

                # Apply local gradients to global network
                global_vars=tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads=trainer.apply_gradients(
                    zip(grads, global_vars))


# In[ ]:


class Worker():
    def __init__(self, game, name, a_size, trainer, model_path, global_episodes, n_obs):
        self.name="worker_" + str(name)
        self.number=name
        self.model_path=model_path
        self.trainer=trainer
        self.global_episodes=global_episodes
        self.increment=self.global_episodes.assign_add(1)
        self.episode_rewards=[]
        self.episode_lengths=[]
        self.episode_mean_values=[]
        self.summary_writer=tf.summary.FileWriter('tfb')

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC=AC_Network(a_size, self.name, trainer, n_obs)
        self.update_local_ops=update_target_graph('global', self.name)
        self.env=game

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout=np.array(rollout)
        states=rollout[:, 0]
        actions=rollout[:, 1]
        rewards=rollout[:, 2]
        timesteps=rollout[:, 3]
        prev_rewards=[0] + rewards[:-1].tolist()
        prev_actions=[0] + actions[:-1].tolist()
        values=rollout[:, 5]

        self.pr=prev_rewards
        self.pa=prev_actions
        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus=np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards=discount(self.rewards_plus, gamma)[:-1]
        self.value_plus=np.asarray(values.tolist() + [bootstrap_value])
        advantages=rewards + gamma * \
            self.value_plus[1:] - self.value_plus[:-1]
        advantages=discount(advantages, gamma)


        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state=self.local_AC.state_init
        feed_dict={self.local_AC.target_v: discounted_rewards,
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
                                              feed_dict = feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, gamma, sess, coord, saver, train):
        episode_count=sess.run(self.global_episodes)
        total_steps=0
        print("start worker:", self.name)

        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer=[]
                episode_values=[]
                episode_reward=0
                episode_step_count=0
                d=False
                r=0
                a=n_actions
                t=0

                env=self.env[np.random.randint(0, len(self.env))]
                sP=env.reset()
                s=processState(sP)
                rnn_state=self.local_AC.state_init

                while episode_step_count < n_actions:
                    # Take an action using probabilities from policy network output.
                    a_dist, v, rnn_state_new=sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                                                        feed_dict = {
                        self.local_AC.state: [s],
                        self.local_AC.prev_rewards: [[r]],
                        self.local_AC.timestep: [[t]],
                        self.local_AC.prev_actions: [a],
                        self.local_AC.state_in[0]: rnn_state[0],
                        self.local_AC.state_in[1]: rnn_state[1]})
                    a=np.random.choice(a_dist[0], p = a_dist[0])
                    a=np.argmax(a_dist == a)

                    rnn_state=rnn_state_new

                    s1P, r, d, obs=env.step(a)
                    s1=processState(s1P)

                    episode_buffer.append([s, a, r, t, d, v[0, 0]])
                    episode_values.append(v[0, 0])
                    episode_reward += r
                    total_steps += 1
                    t += 1
                    episode_step_count += 1
                    s=s1

                    if d == True:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0 and train == True:
                    v_l, p_l, e_l, g_n, v_n=self.train(
                        episode_buffer, sess, gamma, 0.0)

                if episode_count % 1000 == 0 and episode_count != 0:


                    if self.name == 'worker_0':
                        if(save_model == True):
                            if(episode_count % 1000 == 0):
                                mean_reward = np.mean(self.episode_rewards[-1000:])
                                print(episode_count,mean_reward)
                                saver.save(sess, model_path+str(int(mean_reward))+'_'+str(episode_count)+'.cptk')

                        mean_reward = np.mean(self.episode_rewards[-1000:])
                        mean_length = np.mean(self.episode_lengths[-1000:])
                        mean_value = np.mean(self.episode_mean_values[-1000:])
                        summary = tf.Summary()
                        summary.value.add(tag='Perf/Reward',
                                          simple_value=float(mean_reward))
                        if(train == True):

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

                    if(episode_count >= num_episodes):

                        coord.request_stop()

                episode_count += 1



# In[ ]:

gamma = .9  # discount rate for advantage estimation and reward discounting
num_episodes = 80000
train = True
model_path = "./models/temp/"  # The path to save our model to.
env = make_envs1(n=1)[0]
load_model = False
save_model = True
n_obs = env.n_obs
n_actions = env.n_actions
max_epLength = n_actions
a_size = n_actions



# In[ ]:






# In[ ]:


# In[ ]:
train_envs = []
train_envs.append(get_train_envs(get_obs_theta(0,2)))
train_envs.append(get_train_envs(get_obs_theta(1,2)))
train_envs.append(get_train_envs(get_obs_theta(2,2)))
train_envs.append(get_train_envs(get_obs_theta(3,2)))
train_envs.append(get_train_envs(get_obs_theta(4,2)))

t_env = train_envs[4]

if True:
    if load_model==False:

        try:
            shutil.rmtree('./tfb')
        except OSError:
            print ("no dir")

        files = glob.glob(model_path+'*')
        for f in files:
            os.remove(f)

    tf.reset_default_graph()

    with tf.device("/cpu:0"):
        global_episodes = tf.Variable(
            0, dtype=tf.int32, name='global_episodes', trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=3e-4)
        # Generate global network
        master_network = AC_Network(a_size, 'global', None, n_obs)

        # num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
        num_workers = 3
        workers = []
        # Create worker classes
        for i in range(num_workers):
            workers.append(Worker(t_env, i, a_size, trainer,
                                  model_path, global_episodes, n_obs))
        saver = tf.train.Saver(max_to_keep=5)


    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        worker_threads = []
        for worker in workers:
            def worker_work(): return worker.work(gamma, sess, coord, saver, train)
            thread = threading.Thread(target=(worker_work))
            thread.start()
            worker_threads.append(thread)
        coord.join(worker_threads)

# In[ ]:
for k in range(5):
    env_array = get_eval_envs(k)
    print(k)
    if True:
        if True:
            with tf.Session() as sess:

                best_reward = -1000
                ckpt = tf.train.get_checkpoint_state('./models/temp/')
                models = ckpt.all_model_checkpoint_paths
                for i in range(len(models)):
                    str = models[i]
                    re = int(str.split('/')[-1].split('_')[0])
                    if(re>=best_reward):
                        model_path1 = str

                saver.restore(sess,model_path1)


                if True:
                    resum = 0
                    for i in range(len(env_array)):

                        episode_reward = 0
                        episode_step_count = 0
                        d = False
                        r = 0
                        a = n_actions
                        t = 0

                        actions = []
                        env = env_array[i]
                        sP = env.reset()
                        s = processState(sP)
                        rnn_state = master_network.state_init

                        while episode_step_count < max_epLength:
                            # Take an action using probabilities from policy network output.
                            a_dist, v, rnn_state_new = sess.run([master_network.policy, master_network.value, master_network.state_out],
                                                                feed_dict={
                                master_network.state: [s],
                                master_network.prev_rewards: [[r]],
                                master_network.timestep: [[t]],
                                master_network.prev_actions: [a],
                                master_network.state_in[0]: rnn_state[0],
                                master_network.state_in[1]: rnn_state[1]})

                            a = np.random.choice(a_dist[0], p=a_dist[0])
                            a = np.argmax(a_dist == a)

                            rnn_state = rnn_state_new

                            s1P, r, d, obs = env.step(a)
                            s1 = processState(s1P)

                            if(obs == True):

                                s = s1
                                t += 1
                                episode_step_count += 1
                                continue

                            episode_reward += r
                            actions.append(a)
                            episode_step_count += 1
                            s = s1
                            t += 1

                            if d == True:
                                break

                        if(d == False):
                            a = 0
                            actions.append(a)
                            s1, r, d, obs = env.step(a)
                            episode_reward += r

                        #print(i, episode_reward, actions)
                        resum += episode_reward


        print('----------------------------------',resum /len(env_array))

# In[ ]:
'''
train_envs = []
train_envs.append(get_train_envs(get_obs_theta(4,2)))
train_envs.append(get_train_envs(get_obs_theta(3,2)))
train_envs.append(get_train_envs(get_obs_theta(2,2)))

t_env = train_envs[0]
a3c_train(t_env)

# In[ ]:

total_reward = []
for j in range(5):
    print('-',j)
    r=a3c_eval(get_eval_envs(j))
    total_reward.append(r)

'''
# In[ ]:
