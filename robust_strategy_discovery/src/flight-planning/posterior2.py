# In[ ]:

import threading
import multiprocessing
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from PIL import ImageDraw
import matplotlib.cm as cm

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

from matplotlib.backends.backend_pdf import PdfPages


# In[ ]:


def get_outcomes():
    base_variation = np.array([-8,-4,4,8])
    weights = [1/np.sqrt(2),1,np.sqrt(2)]
    all_outcomes = []
    for i in weights:
        for j in range(0,3):
            t=base_variation*weights[j]
            t=np.around(t, decimals=1)
            all_outcomes.append(t)

    outcomes = list(np.unique(all_outcomes))
    return outcomes



# In[ ]:
def get_training_dataset(total_num = 1000):
    combi = list(combinations_with_replacement([0, 1, 2],r=3))
    all = []
    for i in combi:
        all = all+list(permutations(i))
    possiblities = list(set(all))

    prob_array=np.full((1,27),1/27)[0]
    prob_array = prob_array/prob_array.sum()

    that_probs=[]
    for i in range(27):
        target = possiblities[i]

        prob_array1=[]
        temperature = .7
        for set1 in possiblities:
            p = np.exp(-abs(set1[0]-target[0])/temperature)*np.exp(-abs(set1[1]-target[1])/temperature)*np.exp(-abs(set1[2]-target[2])/temperature)
            prob_array1.append(p)
        prob_array1 = np.array(prob_array1)
        prob_array1 = prob_array1/prob_array1.sum()

        that_probs.append(prob_array1)

    outcomes = get_outcomes()

    base_variation = np.array([-8,-4,4,8])
    weights = [1/np.sqrt(2),1,np.sqrt(2)]
    level_poss = [Categorical(np.around(base_variation*weights[0], decimals=1)),Categorical(np.around(base_variation*weights[1], decimals=1)),Categorical(np.around(base_variation*weights[2], decimals=1))]


    theta_set = possiblities

    theta_poss = []
    for i in range(27):
        temp_theta = []
        set1 = theta_set[i]
        for j in range(3):
            temp_theta.append(level_poss[set1[j]])
        theta_poss.append(temp_theta)

    final_theta=[]
    for k in range(27):
        true_thetha = theta_poss[k]
        true_thetha_vector=np.zeros(shape=[len(true_thetha),len(outcomes)])
        for i in range(len(true_thetha)):
            vals = true_thetha[i].vals
            probs = true_thetha[i].probs
            for j in range(len(vals)):
                true_thetha_vector[i][outcomes.index(vals[j])] = probs[j]
        final_theta.append(true_thetha_vector)


    index=np.eye(27)

    all_theta = np.empty((0,27))
    all_that = np.empty((0,30))

    for i in range(27):
        num = prob_array[i]*total_num
        tarr = np.empty((0,30))
        for j in range(27):
            num2 = that_probs[i][j]*num
            that = final_theta[j]
            that =  np.repeat([that.flatten()],num2,axis=0)
            tarr = np.append(tarr,that,axis=0)

        tt = index[i]
        arr=np.array([tt]*tarr.shape[0]).reshape((-1,27))

        all_theta = np.append(all_theta,arr,axis=0)
        all_that = np.append(all_that,tarr,axis=0)


    return all_that,all_theta






# In[ ]:

def train_posterior_function(data,labels):
    path = 'posterior/model2'
    outcomes = get_outcomes()
    def build_model():
      model = keras.Sequential([
        layers.Dense(20, activation='relu', input_shape=[3*len(outcomes)]),
        layers.Dense(10, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(27, activation='softmax'),
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
      epochs=10,
      batch_size=64,
    )

    model.save(path+'.h5')



# In[ ]:

def get_posterior(theta_hat):
    path = 'posterior/model2'+'.h5'
    model = tf.keras.models.load_model(path)
    posterior = model.predict(theta_hat)[0]

    return posterior





# In[ ]:

'''
data,labels = get_training_dataset(total_num = 100000)
train_posterior_function(data,labels)
'''

# In[ ]:

def eval_data(num):

    outcomes = get_outcomes()
    base_variation = np.array([-8,-4,4,8])
    weights = [1/np.sqrt(2),1,np.sqrt(2)]
    data = [Categorical(np.around(base_variation*weights[num[0]], decimals=1)),Categorical(np.around(base_variation*weights[num[1]], decimals=1)),Categorical(np.around(base_variation*weights[num[2]], decimals=1))]


    if True:
        true_thetha = data
        true_thetha_vector=np.zeros(shape=[len(true_thetha),len(outcomes)])
        for i in range(len(true_thetha)):
            vals = true_thetha[i].vals
            probs = true_thetha[i].probs
            for j in range(len(vals)):
                true_thetha_vector[i][outcomes.index(vals[j])] = probs[j]

    return np.array([true_thetha_vector.flatten()])

def eval():

    combi = list(combinations_with_replacement([0, 1, 2],r=3))
    all = []
    for i in combi:
        all = all+list(permutations(i))
    possiblities = list(set(all))



    for i in range(27):
        print(i,possiblities[i])

    path = 'posterior/model2'+'.h5'
    model = tf.keras.models.load_model(path)

    fig = plt.figure(figsize=(15,55))


    for i in range(27):
        data1 = eval_data(possiblities[i])
        posterior = np.around(model.predict(data1)[0],decimals=2)

        ax = fig.add_subplot(27,1,i+1)
        ax.bar(list(range(27)),posterior)
        ax.set_title(str(i)+str(possiblities[i]))
        ax.set_ylim(0,1)

    plt.show()


    pp = PdfPages('foo.pdf')
    pp.savefig(fig)
    pp.close()












# In[ ]:
eval()

# In[ ]:


# In[ ]:

# In[ ]:
