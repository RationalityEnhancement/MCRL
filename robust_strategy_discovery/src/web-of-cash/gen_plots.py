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

breaks = [0, 0, 0, 274, 340, 297, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 323, 0, 0, 379, 0, 0, 376, 0, 0, 345, 0, 0,
 347, 0, 0, 313, 0]

total=0
end_points=[]
for m in range(36):

    total = total+breaks[m]
    end=total
    end_points.append(end)
a = list(set(end_points))
a.sort()
end_points=a[1:]


# In[3]:



# In[3]:
all_resum=[]
temp_sum=0
j=0
c=0
df4=pd.read_csv("csv/temp.csv")['return']
for i in range(end_points[-1]):
    if(i==end_points[j]):
        all_resum.append(temp_sum/c)
        temp_sum=0
        c=0
        j=j+1


    temp_sum=temp_sum+df4.loc[i]
    c=c+1
all_resum.append(temp_sum/c)
print(all_resum)








# In[3]:

def help_fun(arr):
    sum=0
    for k in arr:
        if k>2:
            sum=sum+1
        elif k<-2:
            sum=sum-1

    if(len(arr)==0):
        return 0
    return sum/len(arr)


df4=pd.read_csv("csv/temp.csv")['pmodels']
all_accu=[]
breaks = [  0, 323,   0, 274, 720, 297,   0, 376,   0,   0, 345,   0,   0,  347,   0,   0, 312,   0]
total=0
for m in range(18):
    start = total
    total = total+breaks[m]
    end=total

    accu=[]
    for i in range(start,end):
        t1 = df4.loc[i][1:-1]
        t1 = [float(x.strip()) for x in t1.split(',')]
        accu.append(help_fun(t1[1:]))

    if(start==end):
        all_accu.append(0)
        continue
    all_accu.append(sum(accu)/len(accu))

index = np.arange(18)
ax = plt.subplot(1,1,1)
ax.bar(index,all_accu)
ax.set_title('Process model prediction accuracy in each of the 9 types of env')
plt.xticks(index)

#plt.savefig('tt.pdf')


# In[3]:
breaks = [0, 0, 0, 274, 340, 297, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 323, 0, 0, 379, 0, 0, 376, 0, 0, 345, 0, 0,
 347, 0, 0, 313, 0]
theta_space = [
 [0,0,0],[0,1,1],[0,4,4],
 [1,0,0],[1,1,1],[1,4,4],
 [2,0,0],[2,1,1],[2,4,4],
 [3,0,0],[3,1,1],[3,4,4],
 [4,0,0],[4,1,1],[4,4,4],
 [5,0,0],[5,1,1],[5,4,4],
  [0,0,0],[0,1,1],[0,4,4],
  [1,0,0],[1,1,1],[1,4,4],
  [2,0,0],[2,1,1],[2,4,4],
  [3,0,0],[3,1,1],[3,4,4],
  [4,0,0],[4,1,1],[4,4,4],
  [5,0,0],[5,1,1],[5,4,4]
 ]

n_actions=(10)
index = np.arange(n_actions)
df4=pd.read_csv("csv/m-rl-pmodel.csv")['actions']
m=1
fig = plt.figure(figsize=(15,27))

total=0
count=1
for m in range(36):
    a3c = np.zeros(n_actions)
    start = total
    total = total+breaks[m]
    end=total

    if(start==end):
        continue

    for i in range(start,end):
        t1 = df4.loc[i][1:-1]
        t1 = [int(x.strip()) for x in t1.split(',')]
        for i in t1:
            a3c[i]+=1


    ax = fig.add_subplot(9,1,count)
    ax.bar(index,a3c)
    plt.xticks(index)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    count=count+1

plt.savefig("t.pdf")
# In[3]:
