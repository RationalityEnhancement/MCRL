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
from generate_environments import *



# In[1]:

def optimal_policy(env, return_value=False, verbose=False):
    with Timer() as t:
        Q, V, pi, info = solve(env)
        v = V(env.init)
    if verbose:
        print('optimal -> {:.2f} in {:.3f} sec'.format(v, t.elapsed))

    policy = FunctionPolicy(pi)
    if return_value:
        return policy, V
    else:
        return policy


# In[1]:

def dp_eval(pol,theta_num):

    agent = Agent()
    agent.register(pol)
    n_eval =500
    policy_return=0

    env_array = get_eval_envs(theta_num,n_eval)
    for i in range(n_eval):
        env = env_array[i]
        env.reset()
        agent.register(env)
        tr = agent.run_episode()
        policy_return=policy_return+tr['return']

    return policy_return/n_eval




# In[1]:
env = get_eval_envs(2,1)[0]
env.reset()
pol,V = optimal_policy(env,return_value=True,verbose=True)
print(dp_eval(pol,2))



# In[1]:

env = get_eval_envs(2,5)[2]
agent = Agent()
env.reset()
agent.register(env)
agent.register(pol)
tr = agent.run_episode()
print(tr['actions'])
env.render()
# In[1]:
print(dp_eval(pol, 0))

# In[1]:
