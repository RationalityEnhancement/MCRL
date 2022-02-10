
# In[ ]:


import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
#from tensorflow.examples.tutorials.mnist import input_data
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
from mouselabdiscrete import NewMouselabEnv
from distributions import PiecewiseUniform, expectation
import pandas as pd
#from posterior import get_train_envs
import sys
import pickle
import os

participant_id = int(sys.argv[1])
stock_id = int(sys.argv[2])
dispersion = float(sys.argv[3])
clickcost = int(sys.argv[4])

if dispersion == 0.15:
    dis = 'high'
else:
    dis = 'low'


dir_path = "bias_rewards" + dis + str(clickcost) + "/metarl/"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

from scipy.stats import dirichlet

#training params
gamma = .9999  # discount rate for advantage estimation and reward discounting
#num_episodes = 2000
div = 1

NUM_EPISODES = 100000//div
SAVE_FREQ = 10000//div
SUMMARY_FREQ = 1000//div
NUM_SAMPLES = 4000//div
NUM_TEST_SAMPLES = 4000//div

train = True
save_model = True
load_model = False

biased_file_name = "bias_dists/t_posteriors_" + str(participant_id) + '.pkl'
model_path = "./models/metarl" + dis + str(clickcost) + '/' + str(participant_id) + '_' + str(stock_id) + '/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
csvpath = dir_path + str(participant_id) + '_' + str(stock_id) + '.csv'

gambles = 7
attributes = 4
alpha = dispersion #dispersion
cost=clickcost
repeat_cost = cost * 2

n_actions = attributes + attributes*gambles + 1
max_epLength = n_actions
a_size = n_actions

num_trials = 50

biased_observations = np.array(pickle.load(open(biased_file_name, 'rb'))[stock_id])*num_trials
biased_observations = biased_observations.astype(int)

params_list = []
for i in range(47):
    params_file_name = "indiv_uws_fits_dist/params_" + str(i) + '.pkl'
    bm_params = pickle.load(open(params_file_name, 'rb'))[0]
    params_list.append(bm_params)
    
len_params_list = len(params_list)
prior_alphas = np.array([ 2.04384224,  1.25739246,  2.25103129,  5.28486598, 14.49720341,
       25.89638925, 13.8111208 ,  2.90207659,  0.95780393,  0.95166241])

intervals = [(-100,-80),(-80,-60),(-60,-40),(-40,-20),(-20,0),(0,20),(20,40),(40,60),(60,80),(80,100)]

mhmcmc_iters = 100000//div


# # Metropolis-Hastings MCMC

# In[30]:


epsilon = 0.00001

bins = [(-1, -0.8),
 (-0.8, -0.6),
 (-0.6, -0.4),
 (-0.4, -0.2),
 (-0.2, 0.0),
 (0.0, 0.2),
 (0.2, 0.4),
 (0.4, 0.6),
 (0.6, 0.8),
 (0.8, 1.0)]
bin_reps = [np.round(np.mean(b), 2) for b in bins]

def b_uws(observed_dist):
    x_bar = np.dot(observed_dist, bin_reps)
    x_b = np.abs(bin_reps - x_bar)
    temp = np.multiply(x_b, observed_dist)
    biased_probs = temp/temp.sum()
    return biased_probs

def b_under(observed_dist, k):
    biased_probs = np.array([p**abs(k*b) for p, b in zip(observed_dist, bin_reps)])
    biased_probs = biased_probs/biased_probs.sum()
    return biased_probs/biased_probs.sum()

def bias_model(observed_dist, bm_params):
    p1 = bm_params[0]
    p2 = bm_params[1]
    k = bm_params[2]
    biased_probs = (1-p1)*(1-p2)*observed_dist + p1*(1-p2)*b_uws(observed_dist) + (1-p1)*p2*(b_under(observed_dist, k)) + p1*p2*b_uws(b_under(observed_dist, k))
    return biased_probs

# def bias_model(probs):
#     p1 = bm_params[0]
#     p2 = bm_params[1]
#     k = bm_params[2]
#     k2 = bm_params[3]
#     p3 = 1-p1-p2
#     extreme_probs = np.array([p**(-abs(k*b)) for p, b in zip(probs, bin_reps)])
#     extreme_probs = extreme_probs/extreme_probs.sum()
    
#     forget_probs = np.array([p**abs(k2*b) for p, b in zip(probs, bin_reps)])
#     forget_probs = forget_probs/forget_probs.sum()
    
#     mixture_probs = p1*np.array(probs) + p2*extreme_probs + p3*forget_probs
#     mixture_probs[mixture_probs == 0] = epsilon
    
#     return mixture_probs/mixture_probs.sum()

def dirichlet_proposal(x, scale = 20):
    #qxx = 1
    xnew = dirichlet.rvs(x*scale)[0]
    xnew[xnew == 0.0] = epsilon
    xnew /= np.sum(xnew)
    
    pnew = dirichlet.logpdf(xnew, alpha = x*scale)
    pold = dirichlet.logpdf(x, alpha = xnew*scale)
    qxx = np.exp(pold - pnew)
    #x[x == 0] = epsilon
    #x += x*epsilon
    #return (scipy.stats.dirichlet.rvs((x + epsilon)*scale)[0], qxx)
    return (xnew, qxx)
    
def dirichlet_multinomial(x):
    logprior = dirichlet.logpdf(x, alpha = prior_alphas)
    
    likelihood = 0
    for i in range(len_params_list):
        biased_probs = bias_model(x, params_list[i])
        par_like = scipy.stats.multinomial.pmf(biased_observations, n = num_trials, p = biased_probs)
        likelihood += par_like
    
    likelihood = likelihood/len_params_list
    
    loglikelihood = np.log(likelihood)
    
    return loglikelihood + logprior


# In[31]:


def mh_sampler(x0, post_fn, prop_fn, prop_fn_kwargs={}, iterations=100000):
    """Simple metropolis hastings sampler.
    
    :param x0: Initial array of parameters.
    :param post_fn: Function to compute posterior.
    :param prop_fn: Function to perform jumps.
    :param prop_fn_kwargs: Keyword arguments for proposal function
    :param iterations: Number of iterations to run sampler. Default=100000
    
    :returns: 
        (chain, acceptance, lnprob) tuple of parameter chain , acceptance rate 
        and log-posterior chain.
    """
    
    # number of dimensions
    ndim = len(x0)
    
    # initialize chain, acceptance rate and lnprob
    chain = np.zeros((iterations, ndim))
    lnprob = np.zeros(iterations)
    accept_rate = np.zeros(iterations)
    
    # first samples
    chain[0] = x0
    lnprob0 = post_fn(x0)
    lnprob[0] = lnprob0
    
    # start loop
    naccept = 0
    for ii in range(1, iterations):
        
        # propose
        x_star, factor = prop_fn(x0, **prop_fn_kwargs)
        
        # draw random uniform number
        u = np.random.uniform(0, 1)
        
        # compute hastings ratio
        lnprob_star = post_fn(x_star)
        H = np.exp(lnprob_star - lnprob0) * factor
        
        # accept/reject step (update acceptance counter)
        if u < H:
            x0 = x_star
            lnprob0 = lnprob_star
            naccept += 1
        
        # update chain
        chain[ii] = x0
        lnprob[ii] = lnprob0
        accept_rate[ii] = naccept / ii
        
    return chain, accept_rate, lnprob


import dirichlet as dlt

np.random.seed(10)

#start the mhmcmc chain by the biased observations (+1 so that all bins have non-zero probabilities)
x0 = biased_observations + 1
x0 = x0/np.sum(x0)
chain, ar, lnprob = mh_sampler(x0, dirichlet_multinomial, dirichlet_proposal, 
                                        prop_fn_kwargs={'scale': 250}, iterations = mhmcmc_iters)
#Discard initial samples
burn = int(0.1 * chain.shape[0])
mhmcmc_samples = chain[burn:].copy()

#shuffle the rest of the samples
np.random.shuffle(mhmcmc_samples)

#Use samples for the train and test envs
test_prob_list = mhmcmc_samples[:NUM_TEST_SAMPLES]
train_prob_list = mhmcmc_samples[NUM_TEST_SAMPLES:NUM_SAMPLES + NUM_TEST_SAMPLES]

#Use a dirichlet approx for the posterior distribution
posterior_alphas = dlt.mle(mhmcmc_samples[NUM_SAMPLES + NUM_TEST_SAMPLES:], tol=1e-7, method="fixedpoint")


# In[33]:

probabilities = dirichlet.mean(posterior_alphas)

reward = PiecewiseUniform(intervals, probabilities)

# In[ ]:


#type=0

# In[ ]:


#NUM_WORKERS = 4
#SAVE_MODEL = True  # Change: --> "False"
#LOAD_MODEL = False
#TRAIN = True  # Change: --> "False"


def get_train_envs():
    env_array = []
    np.random.seed(1000)
    for i in range(NUM_SAMPLES):
        true_probs = train_prob_list[i]
        truereward = PiecewiseUniform(intervals, true_probs)
        
        env = NewMouselabEnv(gambles, attributes, reward, cost, ground_truth_dist = truereward, alpha = alpha, sample_term_reward = False)
        env_array.append(env)

    return env_array
# In[ ]:
env_array = get_train_envs()

def make_test_envs():
    test_env_array = []

    np.random.seed(100)
    for i in range(NUM_TEST_SAMPLES):
        test_probs = test_prob_list[i]
        testreward = PiecewiseUniform(intervals, test_probs)
        
        env = NewMouselabEnv(gambles, attributes, reward, cost, ground_truth_dist = testreward, alpha = alpha, sample_term_reward = True)
        test_env_array.append(env)

    return test_env_array


def processState(instate):

    #if isinstance(instate, str):
    #    return None
    if type(instate) == str:
        return None

    mask = np.ones(attributes*gambles + attributes)
    state = np.ones(attributes*gambles + attributes)

    exp_probs = instate[0].expectation()
    for i in range(attributes):
        if instate[0].state[i] < 0:
            mask[i] = -1
            
        state[i] = exp_probs[i]

    for i in range(attributes*gambles):
        if hasattr(instate[1][i], 'sample'):
            mask[i + attributes] = -1
            state[i + attributes] = expectation(instate[1][i])
        else:
            mask[i + attributes] = 1
            state[i + attributes] = instate[1][i]

    mask = mask[:] * 60


    #out = np.stack((state_arr,mask_arr), axis=0)
    out = np.stack((state, mask), axis=1).flatten()
    return out

# In[ ]:

def performAction(env, action):
    possible_actions = env.actions()
    if action in possible_actions:
        s1, r, d, _ = env._step(action)
        return s1, r, d, False

    else:
        if env._state is env.term_state:
            return env._state, 0, True, True
        s1, r, d, _ = env._step(action)
        return s1, -repeat_cost, d, True



# In[ ]:
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


# In[ ]:
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
        self.summary_writer = tf.summary.FileWriter(model_path + 'tfb')

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(a_size, self.name, trainer, n_obs)
        self.update_local_ops = update_target_graph('global', self.name)
        self.env_arr = game

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
                a = -1
                t = 0

                env = self.env_arr[np.random.randint(0, NUM_SAMPLES)]
                sP = env._reset()
                s = processState(sP)
                #s1 = s
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

                    s1P, r, d, _ = performAction(env, a)
                    if not d:
                        s1 = processState(s1P)
                        if s1 is not None:
                            s = s1                    

                    episode_buffer.append([s, a, r, t, d, v[0, 0]])
                    episode_values.append(v[0, 0])
                    episode_reward += r
                    total_steps += 1
                    t += 1
                    episode_step_count += 1                  

                    if d:
                        env._reset()
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0 and train:
                    v_l, p_l, e_l, g_n, v_n = self.train(
                        episode_buffer, sess, gamma, 0.0)

                if save_model:
                    if episode_count % SAVE_FREQ == 0 and episode_count != 0:
                        saver.save(sess, model_path + str(episode_count) + '.cptk')
                        
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



# In[ ]:
# if load_model==False:

#     try:
#         shutil.rmtree('./tfb')
#     except OSError:
#         print ("no dir")

#     files = glob.glob(model_path+'*')
#     for f in files:
#         os.remove(f)

tf.reset_default_graph()

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(
        0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=2e-4)
    # Generate global network
    master_network = AC_Network(a_size, 'global', None, attributes + gambles*attributes)

    #num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
    num_workers = 1
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(env_array, i, a_size, trainer,
                              model_path, global_episodes, attributes + gambles*attributes))
    saver = tf.train.Saver(max_to_keep=10)


# In[ ]:
if(True):
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.all_model_checkpoint_paths[5])
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


env_array = make_test_envs()   

# In[ ]:

#df = pd.DataFrame(columns=['i', 'return', 'actions', 'ground_truth'])
df = pd.DataFrame(columns = ['payoffs', 'probabilities', 'return', 'actions', 'gamble_taken'])
with tf.Session() as sess:


    ckpt = tf.train.get_checkpoint_state(model_path)

    best_reward = -1000
    models = ckpt.all_model_checkpoint_paths
    for i in range(len(models)):
        str = models[i]
        re = int(str.split('/')[-1].split('.')[0])
        if(re>=best_reward):
            model_path = str
            best_reward=re

    #saver.restore(sess, ckpt.all_model_checkpoint_paths[4])
    saver.restore(sess,model_path)

    if True:
        if True:
            resum = 0
            for i in range(len(env_array)):

                episode_reward = 0
                episode_step_count = 0
                d = False
                r = 0
                a = -1
                t = 0

                actions = []
                env = env_array[i]
                sP = env._reset()
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

                    s1P, r, d, obs = performAction(env,a)
                    
                    if not d:
                        s1 = processState(s1P)

                    if(obs == True):

                        s = s1
                        t += 1
                        episode_step_count += 1
                        continue

                    episode_reward += r
                    actions.append(a)
                    episode_step_count += 1
                    
                    t += 1

                    if d == True:
                        break
                    s = s1


                if(d == False):
                    a = attributes + attributes*gambles 
                    actions.append(a)
                    s1, r, d, obs = performAction(env, a)
                    episode_reward += r

                df.loc[i] = (list(env.ground_truth), list(env.distRV.ground_truth), episode_reward, actions, np.argmax(env.mus))
                resum += episode_reward


print(resum / len(env_array))
df.to_csv(csvpath, index = False)


# In[ ]:
'''

df = pd.DataFrame(columns=['i', 'ground_truth'])
with tf.Session() as sess:
    print('Loading Model...')
    #ckpt = tf.train.get_checkpoint_state(model_path)
    #saver.restore(sess, ckpt.all_model_checkpoint_paths[5])
    #saver.restore(sess,ckpt.model_checkpoint_path )
    if True:
        if True:
            resum = 0
            for i in range(len(env_array)):



                env = env_array[i]
                sP = env.reset()
                s = processState(env.ground_truth)

                tt = s[0].flatten()
                t1 = tt[0:3].astype(int)
                t2 = tt[3:6].astype(int)
                t3 = tt[6:].astype(int)
                tt = list(t3)+list(t2)+list(t1)
                print(tt,t1,t2,t3)

                df.loc[i] = [i, tt ]



df.to_csv('csv/temp.csv')
'''
# In[ ]:
