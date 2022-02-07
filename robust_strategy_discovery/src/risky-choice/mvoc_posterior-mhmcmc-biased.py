#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from mouselabdiscrete import NewMouselabEnv
from evaluation import *
from distributions import PiecewiseUniform
from time import time
import sys

# In[2]:


import pickle

participant_id = int(sys.argv[1])
stock_id = int(sys.argv[2])

biased_file_name = "indiv_uws_fits_dist/t_posteriors_" + str(participant_id) + '.pkl'
params_file_name = "indiv_uws_fits_dist/params_" + str(participant_id) + '.pkl'
reward_file_name = "indiv_uws_rewards/" + 'mvocbiased' + str(participant_id) + '_' + str(stock_id) + '.npy'


num_trials = 50

biased_observations = np.array(pickle.load(open(biased_file_name, 'rb'))[stock_id])*num_trials
biased_observations = biased_observations.astype(int)

bm_params = pickle.load(open(params_file_name, 'rb'))[0]


# In[3]:


import scipy.stats
from scipy.stats import dirichlet


gambles = 7
attributes = 4
alpha = 0.15 #dispersion
num_episodes = 500
num_test_episodes = 1000
cost=2

prior_alphas = np.array([ 2.04384224,  1.25739246,  2.25103129,  5.28486598, 14.49720341,
       25.89638925, 13.8111208 ,  2.90207659,  0.95780393,  0.95166241])

intervals = [(-100,-80),(-80,-60),(-60,-40),(-40,-20),(-20,0),(0,20),(20,40),(40,60),(60,80),(80,100)]


# # Metropolis-Hastings MCMC

# In[4]:


epsilon = 0.0001

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

def bias_model(observed_dist):
    p1 = bm_params[0]
    p2 = bm_params[1]
    k = bm_params[2]
    biased_probs = (1-p1)*(1-p2)*observed_dist + p1*(1-p2)*b_uws(observed_dist) + (1-p1)*p2*(b_under(observed_dist, k)) + p1*p2*b_uws(b_under(observed_dist, k))
    return biased_probs


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
    biased_probs = bias_model(x)
    loglikelihood = scipy.stats.multinomial.logpmf(biased_observations, n = num_trials, p = biased_probs)
    
    return loglikelihood + logprior


# In[5]:


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


# In[6]:


import dirichlet as dlt

np.random.seed(10)

#start the mhmcmc chain by the biased observations (+1 so that all bins have non-zero probabilities)
x0 = biased_observations + 1
x0 = x0/np.sum(x0)
chain, ar, lnprob = mh_sampler(x0, dirichlet_multinomial, dirichlet_proposal, 
                                        prop_fn_kwargs={'scale': 240}, iterations = 100000)
#Discard initial samples
burn = int(0.1 * chain.shape[0])
mhmcmc_samples = chain[burn:].copy()

#shuffle the rest of the samples
np.random.shuffle(mhmcmc_samples)

#Use samples for the train and test envs
train_prob_list = mhmcmc_samples[:num_episodes]
test_prob_list = mhmcmc_samples[num_episodes:num_episodes + num_test_episodes]

#Use a dirichlet approx for the posterior distribution
posterior_alphas = dlt.mle(mhmcmc_samples[num_episodes + num_test_episodes:], tol=1e-7, method="fixedpoint")


# In[7]:


ar[-1]


# In[8]:


probabilities = biased_observations/num_trials

reward = PiecewiseUniform(intervals, probabilities)


# In[9]:


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
    
    def voc_estimate(action):
        if action < env.term_action:
            myopic_voc = env.myopic_voi(action)
            
            return myopic_voc + env.cost
            #return w1*features[1] + w2*features[3] + w3*features[2] + w4*features[0]
        else:
            return 0
    
    cumreturn = 0
    unopened = 0
    reward_list = []
    
    for env in test_env_array:
        #print(env.ground_truth)
        exp_return = 0

        while True:
            possible_actions = list(env.actions())

            #take action that maximises estimated VOC
            action_taken = max(possible_actions, key = voc_estimate)
            
            #print(action_taken)
            #if action_taken == env.term_action:
                #print(env._state)
                #print(env.grid())
            _, rew, done, _=env._step(action_taken)
            exp_return+=rew

            if done:
                unopened += len(possible_actions) - 1 
                break
        
        cumreturn += exp_return
        reward_list.append(exp_return)
        #print(exp_return)
    avgclicks = (gambles + 1)*attributes - unopened/num_test_episodes
    np.save(reward_file_name, reward_list)
    print(avgclicks)
    print(cumreturn/num_test_episodes)
    return -cumreturn/num_test_episodes


# In[10]:


testt1 = time()
testfunc()
testt2 = time()


# In[11]:


testt2 - testt1


# In[ ]:




