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
from distributions import PiecewiseUniform, expectation
from mouselabdiscrete import NewMouselabEnv
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
#from posterior import get_train_envs,get_train_envs_that
import sys
import pickle

participant_id = int(sys.argv[1])
stock_id = int(sys.argv[2])
dispersion = float(sys.argv[3])
clickcost = int(sys.argv[4])

if dispersion == 0.15:
    dis = 'high'
else:
    dis = 'low'


dir_path = "bias_rewards" + dis + str(clickcost) + "/drqn/"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

div = 100

# Setting the training parameters
batch_size = 1  # How many experience traces to use for each training step.
update_freq = 5  # How often to perform a training step.
GAMMA = .99  # Discount factor on the target Q-values
startE = 1  # Starting chance of random action
endE = 0.02  # Final chance of random action
# NUM_EPISODES = 20000  # How many steps of training to reduce startE to endE.
annealing_steps = 50000//div
pre_train_steps = 5000//div
#annealing_steps = 400
#pre_train_steps = 200 
learning_rate = 1e-4
time_per_step = 1  # Length of each step used in gif creation
#summaryLength = 100  # Number of epidoes to periodically save for analysis
tau = 0.001
h_size = 300
hidden_size = h_size

NUM_EPISODES = 80000//div
SAVE_FREQ = 5000//div
SUMMARY_FREQ = 1000//div
NUM_SAMPLES = 4000//div
NUM_TEST_SAMPLES = 4000//div

train = True
save_model = True
load_model = False

biased_file_name = "bias_dists/t_posteriors_" + str(participant_id) + '.pkl'
model_path = "./models/drqn" + dis + str(clickcost) + '/' + str(participant_id) + '_' + str(stock_id) + '/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
csvpath = dir_path + str(participant_id) + '_' + str(stock_id) + '.csv'

num_trials = 50

biased_observations = np.array(pickle.load(open(biased_file_name, 'rb'))[stock_id])*num_trials
biased_observations = biased_observations.astype(int)

params_list = []
for i in range(47):
    params_file_name = "indiv_uws_fits_dist/params_" + str(i) + '.pkl'
    bm_params = pickle.load(open(params_file_name, 'rb'))[0]
    params_list.append(bm_params)
    
len_params_list = len(params_list)

from scipy.stats import dirichlet

gambles = 7
attributes = 4
alpha = dispersion #dispersion
cost=clickcost
repeat_cost = cost * 2
n_actions = attributes + attributes*gambles + 1
max_epLength = n_actions

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


# In[32]:


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

#a_size = n_actions


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
        s1, r, d, _ = env._step(action)
        return s1, -repeat_cost, d, True
# In[ ]:

class Qnetwork():
    def __init__(self, h_size, rnn_cell, myScope, n_obs, n_actions, hidden_size):

        self.scalarInput = tf.placeholder(
            shape=[None, n_obs * 2], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 1, n_obs, 2])

        self.conv1 = slim.conv2d(inputs=self.imageIn, num_outputs=36, kernel_size=[1, 8], stride=[
                                 1, 4], padding='VALID', biases_initializer=None, scope=myScope + '_conv1')
        #print(self.conv1)
        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=128, kernel_size=[1, 3], stride=[
                                 1, 3], padding='VALID', biases_initializer=None, scope=myScope + '_conv2')
        #print(self.conv2)
        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=h_size, kernel_size=[1, 2], stride=[
                                 1, 2], padding='VALID', biases_initializer=None, scope=myScope + '_conv3')
        #print(self.conv3)

        self.prev_rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.prev_actions_onehot = tf.one_hot(self.prev_actions, n_actions, dtype=tf.float32)

        #self.hidden = tf.concat([slim.flatten(self.conv3)], 1)
        self.hidden = slim.flatten(self.conv3)

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


def train_dqnrnn(env_array):

    env = env_array[0]
    n_obs = attributes + attributes*gambles
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

#     try:
#         shutil.rmtree('./tfb')
#     except OSError:
#         print ("no dir")

#     files = glob.glob(MODEL_PATH+'*')
#     for f in files:
#         os.remove(f)

    summary_writer = tf.summary.FileWriter(model_path + "tfb")

    with tf.Session() as sess:

        sess.run(init)

        # Set the target network to be equal to the primary network.
        updateTarget(targetOps, sess)
        for i in range(NUM_EPISODES):
            episodeBuffer = []
            # Reset environment and get first new observation
            env = env_array[np.random.randint(0, num_envs)]
            sP = env._reset()
            s = processState(sP)
            d = False
            r = 0
            a = -1
            rAll = 0
            j = 0
            random_ac = np.arange(n_actions)
            np.random.shuffle(random_ac)
            #random_ac = random.sample(range(0,n_actions), n_actions)
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
                    random_ac = random_ac[random_ac != a]

                s1P, r, d, obs = performAction(env, a)
                if not d:
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


                if d == True:
                    break

            # Add the episode to the experience buffer
            bufferArray = np.array(episodeBuffer)

            episodeBuffer = list(zip(bufferArray))
            myBuffer.add(episodeBuffer)
            jList.append(j)
            rList.append(rAll)

            if i % SUMMARY_FREQ == 0 and i!=0:


                mean_reward = np.mean(rList[-SUMMARY_FREQ:])
                summary = tf.Summary()
                summary.value.add(tag='Perf/Reward',
                                  simple_value=float(mean_reward))
                summary_writer.add_summary(summary, i)
                summary_writer.flush()
                print(i,e,mean_reward)

            if save_model:
                if i%SAVE_FREQ==0 and i!=0:
                    saver.save(sess, model_path + str(int(mean_reward))+
                           '_'+str(i)+'.cptk')
            # if(mean_reward<20 and i>15000 and e_increase==False):
            #     e=0.7
            #     e_increase=True

# In[ ]:



# In[ ]:


def dqnrnn_eval(env_array,model_path):
    max_epLength = n_actions
    num_envs = len(env_array)


    tf.reset_default_graph()
    # We define the cells for the primary and target q-networks
    cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=hidden_size, state_is_tuple=True)
    cellT = tf.contrib.rnn.BasicLSTMCell(
        num_units=hidden_size, state_is_tuple=True)
    mainQN = Qnetwork(h_size, cell, 'main', attributes + attributes*gambles, n_actions, hidden_size)
    targetQN = Qnetwork(h_size, cellT, 'target',  attributes + attributes*gambles, n_actions, hidden_size)
    saver = tf.train.Saver(max_to_keep=100)


    test_epLength = n_actions

    df = pd.DataFrame(columns = ['payoffs', 'probabilities', 'return', 'actions', 'gamble_taken'])
    resum = 0

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model_path)

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
            sP = env._reset()
            s = processState(sP)
            actions = []
            d = False
            r=0
            a = -1
            rAll = 0
            j = 0
            state = (np.zeros([1, hidden_size]), np.zeros([1, hidden_size]))
            while j < test_epLength:
                j += 1
                a, state1 = sess.run([mainQN.predict, mainQN.rnn_state],
                                     feed_dict={mainQN.scalarInput: [s], mainQN.trainLength: 1, mainQN.state_in: state, mainQN.batch_size: 1,
                                      mainQN.prev_rewards: [[r]], mainQN.prev_actions: [a]})
                a = a[0]

                s1P, r, d, obs = performAction(env, a)
                if not d:
                    s1 = processState(s1P)

                if(obs==True):
                    s = s1
                    continue
                rAll += r
                actions.append(a)
                if d == True:
                    break
                s = s1
                sP = s1P
                state = state1


            if(d == False):
                a = n_actions - 1

                actions.append(a)
                s1, r, d, obs = performAction(env, a)
                rAll += r

            df.loc[i] = (list(env.ground_truth), list(env.distRV.ground_truth), rAll, actions, np.argmax(env.mus))
            #print(i, rAll, actions)
            resum += rAll

    print('----------------------------',resum /num_envs)
    df.to_csv(csvpath, index = False)
    return resum/num_envs



#env_array,pos = get_train_envs(thatnum)
env_array = get_train_envs()
train_dqnrnn(env_array)

# In[ ]:
env_array = make_test_envs()
dqnrnn_eval(env_array,model_path)

# In[ ]:
