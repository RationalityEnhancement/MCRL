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
from scipy import stats
from helper1 import *
from random import choice
from time import sleep
from time import time
import glob
import shutil
from mouselab import MouselabEnv
from distributions import Categorical, Normal
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import shuffle
from itertools import combinations_with_replacement,permutations


# In[ ]:

def get_outcomes():
    base_variation = np.array([-8,-4,4,8])
    delta_var = [.25,.5,1,2,4]
    all_outcomes = []
    for i in delta_var:
        for j in range(-1,2):
            t=base_variation*pow(i,j)
            all_outcomes.append(t)

    outcomes = list(np.unique(all_outcomes))
    return outcomes



def get_theta(type,num=1000):

    outcomes = get_outcomes()

    base_variation = np.array([-8,-4,4,8])
    delta_var = [.25,.5,1,2,4]
    true_thetha = [Categorical(base_variation*pow(delta_var[type],-1)),
        Categorical(base_variation*pow(delta_var[type],0)),Categorical(base_variation*pow(delta_var[type],1))]

    if(type==0):
        label = np.array([1,0,0,0,0])
    if(type==1):
        label = np.array([0,1,0,0,0])
    if(type==2):
        label = np.array([0,0,1,0,0])
    if(type==3):
        label = np.array([0,0,0,1,0])
    if(type==4):
        label = np.array([0,0,0,0,1])


    true_thetha_vector = np.zeros(shape=[len(true_thetha),len(outcomes)])

    for i in range(len(true_thetha)):
        vals = true_thetha[i].vals
        probs = true_thetha[i].probs
        for j in range(len(vals)):
            true_thetha_vector[i][outcomes.index(vals[j])] = probs[j]


    theta_index = np.array([label]*num).reshape((-1,5))
    theta_vector = np.array([true_thetha_vector]*num)

    return (theta_index,theta_vector,outcomes)
# In[ ]:

def get_thetha_hat(theta,type,process_model_prior=[0.2,0.8,0,0]):
    outcomes =theta[2]
    variation = [[-48, -24, 24, 48],[-8, -4, 4, 8],[-4, -2, 2, 4]]
    total_num = theta[0].shape[0]
    numP1 = int(total_num*(process_model_prior[0]))
    numP2 = int(total_num*(process_model_prior[1]))
    numP3 = int(total_num*(process_model_prior[2]))
    numP4 = int(total_num*(process_model_prior[3]))

    theta_hatP1 = np.empty((0,3,len(outcomes)))
    theta_hatP2 = np.empty((0,3,len(outcomes)))
    theta_hatP3 = np.empty((0,3,len(outcomes)))
    theta_hatP4 = np.empty((0,3,len(outcomes)))

    process_label_P1 = np.empty((0,4))
    process_label_P2 = np.empty((0,4))
    process_label_P3 = np.empty((0,4))
    process_label_P4 = np.empty((0,4))



    mistake_prob = [[2/3,1/3,0,0,0],[1/4,1/2,1/4,0,0],[0,1/4,1/2,1/4,0],[0,0,1/4,1/2,1/4],[0,0,0,1/3,2/3]]


    if (numP1!=0):
        mu=0
        #mu = (35/3)/10
        n_samples = 10
        gamma = 1.5
        true_thetha_vector = theta[1][0]

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
        for j in range(numP1):
            single_theta_hat = []
            for i in range(3):
                ind = np.random.choice(len(outcomes),n_samples,p=w[i])
                res = stats.relfreq(ind, len(outcomes),defaultreallimits=(0,len(outcomes)))
                single_theta_hat.append(list(res[0]))


            theta_hat.append(single_theta_hat)

        theta_hatP1=np.array(theta_hat)
        process_label_P1 = np.array([[1,0,0,0]]*numP1)


    if (numP2!=0):
        for i in range(5):
            t = get_theta(i,num=int(mistake_prob[type][i]*numP2))[1]
            if(t.shape[0]==0):
                continue
            theta_hatP2=np.append(theta_hatP2,t,axis=0)

        process_label_P2 = np.array([[0,1,0,0]]*numP2)



    theta_hat = np.concatenate((theta_hatP1,theta_hatP2,theta_hatP3,theta_hatP4),axis=0)
    process_labels = np.concatenate((process_label_P1,process_label_P2,process_label_P3,process_label_P4),axis=0)

    if(theta_hat.shape[0]!=total_num):
        tnum = total_num - theta_hat.shape[0]
        t = get_theta(3,num=tnum)[1]
        theta_hat=np.append(theta_hat,t,axis=0)
    if(process_labels.shape[0]!=total_num):
        tnum = total_num - process_labels.shape[0]
        t=np.array([[1,0,0,0]]*tnum)
        process_labels = np.append(process_labels,t,axis=0)


    return theta_hat,process_labels

# In[ ]:

def get_training_dataset(num_samples=50000):

    prior_dist = [0.1,0.1,0.1,0.5,0.2]
    outcomes = get_outcomes()

    theta0 = get_theta(0,num=int(num_samples*prior_dist[0]))
    theta_hat0,pLabels0 = get_thetha_hat(theta0,0)
    theta1 = get_theta(1,num=int(num_samples*prior_dist[1]))
    theta_hat1,pLabels1 = get_thetha_hat(theta1,1)
    theta2 = get_theta(2,num=int(num_samples*prior_dist[2]))
    theta_hat2,pLabels2 = get_thetha_hat(theta2,2)
    theta3 = get_theta(3,num=int(num_samples*prior_dist[3]))
    theta_hat3,pLabels3 = get_thetha_hat(theta3,3)
    theta4 = get_theta(4,num=int(num_samples*prior_dist[4]))
    theta_hat4,pLabels4 = get_thetha_hat(theta4,4)


    data = np.concatenate((theta_hat0,theta_hat1,theta_hat2,theta_hat3,theta_hat4),axis=0)
    data = data.reshape((-1,3*len(outcomes)))
    labels = np.concatenate((theta0[0],theta1[0],theta2[0],theta3[0],theta4[0]),axis=0)
    process_labels = np.concatenate((pLabels0,pLabels1,pLabels2,pLabels3,pLabels4),axis=0)
    data,labels,process_labels = shuffle(data,labels,process_labels)

    return (data,labels,process_labels)


# In[ ]:

def train_posterior_function(data,labels):
    path = 'posterior/model'
    outcomes = get_outcomes()
    def build_model():
      model = keras.Sequential([
        layers.Dense(20, activation='relu', input_shape=[3*len(outcomes)]),
        layers.Dense(10, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(5, activation='softmax'),
      ])

      return model

    model = build_model()

    model.compile(
      optimizer='adam',
      loss='categorical_crossentropy',
    )


    model.fit(
      data, # training data
      labels, # training targets
      epochs=5,
      batch_size=64,
    )

    model.save(path+'.h5')



# In[ ]:

def get_posterior(theta_hat):
    path = 'posterior/model'+'.h5'
    model = tf.keras.models.load_model(path)
    posterior = model.predict(theta_hat)[0]

    return posterior


# In[ ]:


def train_process_posterior_function(data,labels):
    path = 'posterior/process_model'
    outcomes = get_outcomes()
    def build_model():
      model = keras.Sequential([
        layers.Dense(20, activation='relu', input_shape=[3*len(outcomes)]),
        layers.Dense(10, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(4, activation='softmax'),
      ])

      return model

    model = build_model()

    model.compile(
      optimizer='adam',
      loss='categorical_crossentropy',
    )


    model.fit(
      data, # training data
      labels, # training targets
      epochs=5,
      batch_size=64,
    )

    model.save(path+'.h5')


# In[ ]:
def get_process_posterior(theta_hat):
    path = 'posterior/process_model'+'.h5'
    model = tf.keras.models.load_model(path)
    posterior = model.predict(theta_hat)[0]

    return posterior


# In[ ]:
def eval(pmodel,num):
    outcomes = get_outcomes()
    if(pmodel==1):
        theta = get_theta(num,1)[1][0]

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

        fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(10,8))

        posterior_dist = get_posterior(theta_hat)
        print("posterior_dist:",posterior_dist)
        ax1.bar([0,1,2,3,4],posterior_dist)
        ax1.set_title('Theta_posterior')


        posterior_dist = get_process_posterior(theta_hat)
        print("process_model_posterior:",posterior_dist)
        ax2.bar([1,2,3,4],posterior_dist)
        ax2.set_title('Process_model_posterior')

        plt.show()



    if(pmodel==2):
        theta_hat = get_theta(num,1)[1][0].reshape((-1,3*len(outcomes)))

        fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(10,8))
        posterior_dist = get_posterior(theta_hat)
        print("posterior_dist:",posterior_dist.shape)
        ax1.bar([0,1,2,3,4],posterior_dist)
        ax1.set_title('Theta_posterior')


        posterior_dist = get_process_posterior(theta_hat)
        print("get_process_posterior:",posterior_dist)
        ax2.bar([1,2,3,4],posterior_dist)
        ax2.set_title('Process_model_posterior')

        plt.show()


# In[ ]:

'''
(data,labels,process_labels) = get_training_dataset()
train_posterior_function(data,labels)
train_process_posterior_function(data,process_labels)

'''


# In[ ]:


combi = list(combinations_with_replacement([0, 1, 2],r=3))
all = []
for i in combi:
    all = all+list(permutations(i))
possiblities = list(set(all))


# In[ ]

prob_array=[]
temperature = .7
for set in possiblities:
    p = np.exp(-abs(set[0]-1)/temperature)*np.exp(-abs(set[1]-1)/temperature)*np.exp(-abs(set[2]-1)/temperature)
    prob_array.append(p)
prob_array = np.array(prob_array)
prob_array = prob_array/prob_array.sum()
plt.bar(list(range(27)),prob_array)



# In[ ]
prob_array=[]
array = [1/np.sqrt(2),1,np.sqrt(2)]
temperature = .3
for set in possiblities:
    p = np.exp(-abs(array[set[1]]-array[set[0]])/temperature)*np.exp(-abs(array[set[2]]-array[set[1]])/temperature)
    prob_array.append(p)
prob_array=np.array(prob_array)
prob_array = prob_array/prob_array.sum()
plt.bar(list(range(27)),prob_array)

# In[ ]:
for i in range(27):
    print(i,possiblities[i],prob_array[i])

# In[ ]:
