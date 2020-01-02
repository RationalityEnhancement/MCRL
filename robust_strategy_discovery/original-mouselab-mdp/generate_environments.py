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
from posterior import *


# In[ ]:

def get_train_envs(theta_hat,total_envs = 1000):
    cost=1
    repeat_cost = 2

    posterior_dist = get_posterior(theta_hat)
    num_envs = []
    for i in range(posterior_dist.shape[0]):
        num_envs.append(int(posterior_dist[i]*total_envs))

    num_envs = np.array(num_envs)
    if(num_envs.sum()!=total_envs):
        num_envs[0] = num_envs[0]+(total_envs-num_envs.sum())

    print(num_envs)

    sample_envs = []
    for i in range(posterior_dist.shape[0]):
        sample_envs.append(get_theta(i,1)[1][0])

    env_array = []
    outcomes = get_outcomes()

    for i in range(posterior_dist.shape[0]):

        def reward(depth):
            if depth == 1:
                return Categorical(list(np.take(outcomes,np.nonzero(sample_envs[i][0]))[0]))
            elif depth == 2:
                return Categorical(list(np.take(outcomes,np.nonzero(sample_envs[i][1]))[0]))
            elif depth == 3:
                return Categorical(list(np.take(outcomes,np.nonzero(sample_envs[i][2]))[0]))
            else:
                return 0

        branching = [4, 1, 1]
        env_array=env_array+[MouselabEnv.new_symmetric(branching, reward, seed=j, cost=cost, sample_term_reward=False, env_type='new',repeat_cost=repeat_cost) for j in range(num_envs[i])]


    return env_array


# In[ ]:
def get_obs_theta(theta_num,pmodel):
    outcomes = get_outcomes()
    theta_hat=[]
    if(pmodel==1):
        theta = get_theta(theta_num,1)[1][0]

        mu=0
        n_samples = 10
        gamma = 1.5
        true_thetha_vector = theta

        o_max = max(outcomes)
        o_min = min(outcomes)

        #calculate utility matrix
        u = np.array(outcomes)
        u = u/(o_max-o_min)


        u = np.array([u,u,u])
        u_mu = mu/(o_max-o_min)
        u = u-u_mu

        p_dash = np.multiply(true_thetha_vector,np.abs(u))

        w_dash = np.power(p_dash,gamma)/np.power((np.power(p_dash,gamma)+np.power(1-p_dash,gamma)),1/gamma)

        w=w_dash/w_dash.sum(axis=1)[:,None]

        theta_hat=[]
        for j in range(1):
            single_theta_hat = []
            for i in range(3):
                ind = np.random.choice(len(outcomes),n_samples,p=w[i])
                res = stats.relfreq(ind, len(outcomes),defaultreallimits=(0,len(outcomes)))
                single_theta_hat.append(list(res[0]))


            theta_hat.append(single_theta_hat)
            theta_hat=np.array(theta_hat)

        theta_hat = theta_hat[0]

        theta_hat = theta_hat.reshape((-1,3*len(outcomes)))

    if(pmodel==2):
        theta_hat = get_theta(theta_num,1)[1][0].reshape((-1,3*len(outcomes)))

    return theta_hat

# In[ ]:
def get_eval_envs(theta_num,num=500):

    cost=1
    repeat_cost = 0

    sample_env = get_theta(theta_num,1)[1][0]

    env_array = []
    outcomes = get_outcomes()

    if True:

        def reward(depth):
            if depth == 1:
                return Categorical(list(np.take(outcomes,np.nonzero(sample_env[0]))[0]))
            elif depth == 2:
                return Categorical(list(np.take(outcomes,np.nonzero(sample_env[1]))[0]))
            elif depth == 3:
                return Categorical(list(np.take(outcomes,np.nonzero(sample_env[2]))[0]))
            else:
                return 0

        branching = [4, 1, 1]
        env_array=[MouselabEnv.new_symmetric(branching, reward, seed=j, cost=cost, sample_term_reward=False, env_type='new',repeat_cost=repeat_cost) for j in range(num)]


    return env_array




# In[ ]:





# In[ ]:
