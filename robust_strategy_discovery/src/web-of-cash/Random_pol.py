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
def random_eval(theta_num):
    n_eval =500
    policy_return=0

    env_array = get_eval_envs(theta_num,n_eval)


    test_epLength = 13



    with tf.Session() as sess:
        reward_total=0

        for i in range(n_eval):
            random_ac = random.sample(range(0,test_epLength), test_epLength)
            env=env_array[i]
            sP = env.reset()
            s = sP
            actions=[]
            d = False
            rAll = 0
            j=1

            while j < test_epLength:

                j+=1
                a = random_ac[0]
                random_ac = random_ac[1:]

                s1,r,d,obs = env.step(a)

                if(obs==True):
                    r=0
                rAll += r
                actions.append(a)

                if d == True:
                    break

            if(d==False):
                a = 0
                actions.append(a)
                s1,r,d,obs = env.step(a)
                rAll += r

            #print(i,rAll,actions)
            reward_total = reward_total+rAll

    return reward_total/n_eval



# In[1]:
print(random_eval(3))

# In[1]:
