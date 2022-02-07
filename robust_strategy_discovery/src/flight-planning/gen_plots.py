#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
import csv
import itertools
import tensorflow.contrib.slim as slim
from mouselab import MouselabEnv
from distributions import Categorical, Normal
from policies import FunctionPolicy
from exact import solve
from contexttimer import Timer
from agents import Agent
import pandas as pd

from helper import *


# In[3]:

n_actions=(13)
index = np.arange(n_actions+1)
df4=pd.read_csv("csv/a3c1.csv")['actions']
a3c = np.zeros(n_actions+1)
for i in range(500):
    t1 = df4.loc[i][1:-1]
    t1 = [int(x.strip()) for x in t1.split(',')]
    for i in t1:
        a3c[i]+=1


ax = plt.subplot(1,1,1)
ax.bar(index,a3c)
ax.set_title('a3c')
plt.xticks(index)
plt.savefig('env2')
# In[3]:

# In[3]:


df1=pd.read_csv("csv/optimal.csv")['return']
df2=pd.read_csv("csv/dqn-rnn.csv")['return']
df3=pd.read_csv("csv/random.csv")['return']
df4=pd.read_csv("csv/a3c.csv")['return']


total_dqn=0
total_a3c=0
total_dqrnn=0
total_random=0
for i in range(500):
    diff1 = df2.loc[i]-df1.loc[i]
    total_dqrnn+=diff1

    diff2 = df3.loc[i]-df1.loc[i]
    total_random+=diff2

    diff3 = df4.loc[i]-df1.loc[i]
    total_a3c+=diff3



print("DQRNN-OPTIMAL: ",total_dqrnn/500)
print("RANDOM-OPTIMAL: ",total_random/500)
print("A3C-OPTIMAL: ",total_a3c/500)





# In[5]:


n_actions=(13)
df1=pd.read_csv("csv/optimal.csv")['actions']
df2=pd.read_csv("csv/dqn-rnn.csv")['actions']
df3=pd.read_csv("csv/random.csv")['actions']
df4=pd.read_csv("csv/a3c.csv")['actions']


index = np.arange(n_actions+1)
optimal = np.zeros(n_actions+1)
dqrnn = np.zeros(n_actions+1)
dqn = np.zeros(n_actions+1)
a3c = np.zeros(n_actions+1)
random = np.zeros(n_actions+1)

for i in range(500):
    t1 = df1.loc[i][1:-1]
    t1 = [int(x.strip()) for x in t1.split(',')]
    for i in t1:
        optimal[i]+=1

for i in range(500):
    t1 = df2.loc[i][1:-1]
    t1 = [int(x.strip()) for x in t1.split(',')]
    for i in t1:
        dqrnn[i]+=1


for i in range(500):
    t1 = df3.loc[i][1:-1]
    t1 = [int(x.strip()) for x in t1.split(',')]
    for i in t1:
        random[i]+=1

for i in range(500):
    t1 = df4.loc[i][1:-1]
    t1 = [int(x.strip()) for x in t1.split(',')]
    for i in t1:
        a3c[i]+=1



fig = plt.figure(figsize=(12,18))
ax = plt.subplot(5,1,1)
ax.bar(index,optimal)
ax.set_title('OPTIMAL')
plt.xticks(index)
ax = plt.subplot(5,1,2)
ax.set_title('DQRNN')
ax.bar(index,dqrnn)
plt.xticks(index)

ax = plt.subplot(5,1,3)
ax.bar(index,random)
ax.set_title('RANDOM')
plt.xticks(index)

ax = plt.subplot(5,1,4)
ax.bar(index,a3c)
ax.set_title('a3c')
plt.xticks(index)

#plt.savefig('compare-model')
plt.show()



# In[]:














# In[]:


test_epLength = n_actions
df = pd.DataFrame(columns=['i','return','actions','ground_truth'])


with tf.Session() as sess:
    sess.run(init)

    for i in range(500):
        env=env_array2[i]
        sP = env.reset()
        s = sP
        actions=[]
        d = False
        rAll = 0
        j=0

        while j < test_epLength:
            j+=1
            a = np.random.randint(0,n_actions)

            s1,r,d,obs = env.step(a)

            if(r==-100):
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

        df.loc[i] = [i,rAll,actions,env.ground_truth]

df.to_csv('random.csv')
