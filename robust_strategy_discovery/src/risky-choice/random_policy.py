#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
from mouselabdiscrete import NewMouselabEnv
from evaluation import *
from distributions import PiecewiseUniform, expectation
from time import time

from joblib import Parallel, delayed
import multiprocessing
import os

num_cores = multiprocessing.cpu_count()

import pandas as pd

import sys
# In[28]:


import pickle

participant_id = int(sys.argv[1])
stock_id = int(sys.argv[2])
dispersion = float(sys.argv[3])
clickcost = int(sys.argv[4])

if dispersion == 0.15:
    dis = 'high'
else:
    dis = 'low'


dir_path = "bias_rewards" + dis + str(clickcost) + "/random/"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

biased_file_name = "bias_dists/t_posteriors_" + str(participant_id) + '.pkl'
csv_file_name = dir_path + str(participant_id) + '_' + str(stock_id) + '.csv'

num_trials = 50

biased_dist = np.array(pickle.load(open(biased_file_name, 'rb'))[stock_id])
biased_observations = biased_dist*num_trials
biased_observations = biased_observations.astype(int)

params_list = []
for i in range(47):
    params_file_name = "indiv_uws_fits_dist/params_" + str(i) + '.pkl'
    bm_params = pickle.load(open(params_file_name, 'rb'))[0]
    params_list.append(bm_params)
    
len_params_list = len(params_list)


# In[29]:


import scipy.stats
from scipy.stats import dirichlet

div = 500

gambles = 7
attributes = 4
alpha = dispersion #dispersion
num_episodes = 500//div
num_test_episodes = 4000//div
cost=clickcost

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

#Use samples for the test envs
test_prob_list = mhmcmc_samples[:num_test_episodes]

# In[33]:


#probabilities = dirichlet.mean(posterior_alphas)

reward = PiecewiseUniform(intervals, biased_dist)


# In[34]:

# In[35]:

# In[36]:

test_env_array = []

def make_test_envs(num_test_episodes = num_test_episodes):
    np.random.seed(100)
    for i in range(num_test_episodes):
        test_probs = test_prob_list[i]
        testreward = PiecewiseUniform(intervals, test_probs)
        
        env = NewMouselabEnv(gambles, attributes, reward, cost, ground_truth_dist = testreward, alpha = alpha, sample_term_reward = True)
        test_env_array.append(env)

make_test_envs()   

def testfunc(num_test_episodes = num_test_episodes):
    def run_episode(env):
    #for env in test_env_array:
        #print(env.ground_truth)
        env._reset()
        exp_return = 0
        action_list = []
        while True:
            possible_actions = list(env.actions())

            #take random action
            action_taken = np.random.choice(possible_actions)
           
            _, rew, done, _=env._step(action_taken)
            
            exp_return+=rew

            if done:
                #unopened += len(possible_actions) - 1 
                break
            
            action_list.append(action_taken)
            
        return (list(env.ground_truth), list(env.distRV.ground_truth), exp_return, action_list, np.argmax(env.mus))
                 
    info_list = Parallel(n_jobs=num_cores)(delayed(run_episode)(env) for env in test_env_array)
    
    df = pd.DataFrame(info_list, columns = ['payoffs', 'probabilities', 'return', 'actions', 'gamble_taken'])
    
    cumreturn = np.sum(df['return'])
    #print(exp_return)
    #avgclicks = (gambles + 1)*attributes - unopened/num_test_episodes
    #print(avgclicks)
    print(cumreturn/num_test_episodes)
    df.to_csv(csv_file_name, index = False)
    #np.save(reward_file_name, reward_list)
    return -cumreturn/num_test_episodes


# In[50]:

testfunc()

# In[51]:


# In[52]:


#posterior_alphas


# In[ ]:




