#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import pandas as pd
from mouselabdiscrete import NewMouselabEnv
from evaluation import *
from distributions import sample, expectation, Normal, Categorical, Mixture, PiecewiseUniform
import torch.nn as nn
import torch
from time import time


# In[28]:


import pickle

participant_id = int(sys.argv[1])
stock_no = int(sys.argv[2])

biasedfilename = 'model_fits_dist/' + str(participant_id) + '_biased_probs.pkl'
biasedfile = open(biasedfilename,'rb')
biaseddists = pickle.load(biasedfile)

truefilename = 'model_fits_dist/' + str(participant_id) + '_true_distributions.pkl'
truefile = open(truefilename,'rb')
truedists = pickle.load(truefile)
#pickle.load('model_fits_dist/0_biased_probs.pkl')


# In[29]:


biased_probs = biaseddists[sorted(biaseddists.keys())[stock_no]]

true_probs = truedists[sorted(truedists.keys())[stock_no]]

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


# In[30]:


num_datapoints = 10000
num_epochs = 5000
num_test_episodes = 1000


# In[31]:


gambles = 7
attributes = 4
scaledist = 100
scalecost = 200
#low_stakes = Normal((0.25+0.01)/2, 0.3*(0.25-0.01))
reward = PiecewiseUniform(bins, biased_probs)*scaledist
testreward = PiecewiseUniform(bins, true_probs)*scaledist
cost=0.01*scalecost

alpha = 0.15

max_action = (gambles + 1)*attributes + 1


# In[32]:


rewardfilename = 'biasedrewards/mvoc' + str(scalecost) + '_' + str(participant_id) + '_' + str(stock_no) + '.npy'


# In[33]:


class BMPS_Approximator(nn.Module):
    def __init__(self, input_size, output_size):
        super(BMPS_Approximator, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.network = nn.Sequential(
                            nn.Linear(input_size, input_size//2),
                            nn.ReLU(),
                            nn.Linear(input_size // 2, input_size // 4),
                            nn.ReLU(),
                            nn.Linear(input_size // 4, input_size//8),
                            nn.ReLU(),
                            nn.Linear(input_size // 8, output_size)
                            )
    def forward(self, X):
        output = self.network(X)
        return output
    
    def train_epoch(self, train_X, train_Y, criterion, optimizer):
        output = self.forward(train_X)
        loss = criterion(output, train_Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

def train_model(model, train_X, train_Y, criterion, optimizer, num_epochs=num_epochs):
    for epoch_num in range(1, num_epochs+1):
        if epoch_num % 1000 == 0:
            print(f"Epoch {epoch_num}")
        loss = model.train_epoch(train_X, train_Y, criterion, optimizer)
        if epoch_num % 1000 == 0:
            print(loss.item())


# In[34]:


def create_tensors(feature_df, vpi=False):
    X = []
    Y = []
    for index, data in feature_df.iterrows():
        #x = []
        y = []
        x = data['beliefstate']
        if not vpi:
            x = np.concatenate((x, data['one_hot_action']))
        for f in ['expected_term_reward', 'cost']:
            x = np.append(x, data[f])
        #for f in features:
        #    x.append(data[f])
        for f in ['myopic_voc']:
            y.append(data[f])
        X.append(x)
        Y.append(y)
    return torch.tensor(X, requires_grad=True), torch.tensor(Y, requires_grad=True)


# In[35]:


def gen_data(num_points, seed = None):
    if seed is not None:
        np.random.seed(seed)
    for i in range(num_points):
        env = NewMouselabEnv(gambles, attributes, reward, cost, alpha = alpha)
        
        #exclude terminal action while devising current state
        possible_actions = list(env.actions())[:-1]
        
        num_attributes = np.random.choice(attributes)
        num_actions = np.random.choice(attributes*gambles)
        
        attributes_taken = np.random.choice(possible_actions[:attributes], size = num_attributes, replace = False)
        actions_taken = np.random.choice(possible_actions[attributes:], size = num_actions, replace = False)

        actions_taken = np.concatenate((attributes_taken, actions_taken))
        
        for action in actions_taken:
            env._step(action)
        
        a = np.array(list(env.actions()))
        possible_actions = list(a[a < attributes])
        
        
        #possible_actions.append(env.term_action)
        
        action = np.random.choice(possible_actions)
        feats = np.array([
                env.cost,
                env.myopic_voi(action),
                env.expected_term_reward()
            ])
        
        state = np.vectorize(lambda g: expectation(g), otypes = [float])(env._state[1])
        
        #print(env.mus, env.vars)
        gamble_feats = env.mus
        
        yield (np.concatenate((env.dist, gamble_feats, state)), np.sort(actions_taken), action, *feats)


# In[36]:


def get_one_hot_encoding(row):
    one_hot_action = np.zeros(max_action)
    one_hot_action[row.actions_taken] = 1
    one_hot_action[row.action] = 2
    return one_hot_action
    


# In[37]:


t1 = time()


# In[38]:


data = list(gen_data(num_datapoints))


# In[39]:


df = pd.DataFrame(data, columns = ['beliefstate', 'actions_taken', 'action', 'cost','myopic_voc', 'expected_term_reward'])


# In[40]:


df['one_hot_action'] = df.apply(get_one_hot_encoding, axis=1)


# In[41]:


X, Y = create_tensors(df)


# In[42]:


myopic_voc_approx = BMPS_Approximator(X.shape[-1], 1)


# In[43]:


# Hyperparam
learning_rate = 1e-4

criterion = nn.MSELoss()
# Optimizers
mvoc_optimizer = torch.optim.Adam(myopic_voc_approx.parameters(), lr=learning_rate)

# Train the networks
train_model(myopic_voc_approx, X.float(), Y[:, 0].unsqueeze_(1), criterion, mvoc_optimizer)

#create_dir("voc_models")
#torch.save(myopic_voc_approx.state_dict(), "voc_models/mvoc_attributesbins.pth")


# In[44]:


t2 = time()


# In[45]:


t2 - t1


# ---

# ## Testing

# In[46]:


sample_term_reward = True


# In[47]:


def testsamefunc():
    
    #num_episodes = 5
    def voc_estimate(action):
        if action < attributes:
            state = np.vectorize(lambda g: expectation(g), otypes = [float])(env._state[1])
            gamble_feats = env.mus
            vpi_x = np.concatenate((env.dist, gamble_feats, state))
        
            one_hot_action = np.zeros(max_action)
            one_hot_action[actions_taken] = 1
            one_hot_action[action] = 2
        
            term_reward = env.expected_term_reward()
            x = np.concatenate((vpi_x, one_hot_action))
            x = np.append(x, [term_reward, env.cost])
        
            X = torch.Tensor([x])
        
            myopic_voc = myopic_voc_approx(X)[0].item()

            return myopic_voc + env.cost
        
        elif action < env.term_action:
            
            myopic_voc = env.myopic_voi(action)
            return myopic_voc + env.cost
            #return w1*features[1] + w2*features[3] + w3*features[2] + w4*features[0]
        else:
            return 0
    
    cumreturn = 0
    
    np.random.seed(100)
    
    for i in range(num_test_episodes):
        env = NewMouselabEnv(gambles, attributes, reward, cost, alpha=alpha, sample_term_reward = sample_term_reward)
    #for env in env_array:

        exp_return = 0
        actions_taken = []
        
        while True:
            possible_actions = list(env.actions())

            #take action that maximises estimated VOC
            action_taken = max(possible_actions, key = voc_estimate)

            _, rew, done, _=env._step(action_taken)
            
            exp_return+=rew
            actions_taken.append(action_taken)
            
            if done:
                #env._reset()
                break
        
        cumreturn += exp_return
        #print(exp_return)
    
    print(cumreturn/num_test_episodes)
    return -cumreturn/num_test_episodes


# In[48]:


#testsamefunc()


# In[49]:


def testfunc():
    
    #num_episodes = 5
    
    
    def voc_estimate(action):
        if action < attributes:
            state = np.vectorize(lambda g: expectation(g), otypes = [float])(env._state[1])
            gamble_feats = env.mus
            vpi_x = np.concatenate((env.dist, gamble_feats, state))
        
            one_hot_action = np.zeros(max_action)
            one_hot_action[actions_taken] = 1
            one_hot_action[action] = 2
        
            term_reward = env.expected_term_reward()
            x = np.concatenate((vpi_x, one_hot_action))
            x = np.append(x, [term_reward, env.cost])
        
            X = torch.Tensor([x])
            VPI_X = torch.Tensor([vpi_x])
        
            myopic_voc = myopic_voc_approx(X)[0].item()
        
            return myopic_voc + env.cost
        
        elif action < env.term_action:
            
            myopic_voc = env.myopic_voi(action)
            
            return myopic_voc + env.cost
            #return w1*features[1] + w2*features[3] + w3*features[2] + w4*features[0]
        else:
            return 0
    
    cumreturn = 0
    
    np.random.seed(100)
    unopened = 0
    rewardlist = []
    for i in range(num_test_episodes):
        env = NewMouselabEnv(gambles, attributes, reward, cost, ground_truth_dist = testreward, alpha=alpha, sample_term_reward = sample_term_reward)
    #for env in env_array:

        exp_return = 0
        actions_taken = []
        
        while True:
            possible_actions = list(env.actions())

            #take action that maximises estimated VOC
            action_taken = max(possible_actions, key = voc_estimate)

            _, rew, done, _=env._step(action_taken)
            
            exp_return+=rew
            actions_taken.append(action_taken)
            
            if done:
                unopened += len(possible_actions) - 1 
                #env._reset()
                break
        
        cumreturn += exp_return
        rewardlist.append(exp_return)
        #print(exp_return)
    
    avgclicks = (gambles + 1)*attributes - unopened/num_test_episodes
    print(avgclicks)
    print(cumreturn/num_test_episodes)
    np.save(rewardfilename, rewardlist)
    return -cumreturn/num_test_episodes


# In[50]:


testfunc()


# In[51]:


def observe_strategy(num_episodes = 5):
    
    #num_episodes = 5
    def voc_estimate(action):
        if action < attributes:
            state = np.vectorize(lambda g: expectation(g), otypes = [float])(env._state[1])
            gamble_feats = env.mus
            vpi_x = np.concatenate((env.dist, gamble_feats, state))
        
            one_hot_action = np.zeros(max_action)
            one_hot_action[actions_taken] = 1
            one_hot_action[action] = 2
        
            term_reward = env.expected_term_reward()
            x = np.concatenate((vpi_x, one_hot_action))
            x = np.append(x, [term_reward, env.cost])
        
            X = torch.Tensor([x])
        
            myopic_voc = myopic_voc_approx(X)[0].item()
        
            return myopic_voc + env.cost
        
        elif action < env.term_action:
            
            myopic_voc = env.myopic_voi(action)
            
            return myopic_voc + env.cost
            #return w1*features[1] + w2*features[3] + w3*features[2] + w4*features[0]
        else:
            return 0
    
    cumreturn = 0
    
    np.random.seed(100)
    unopened = 0
    rewardlist = []
    for i in range(num_episodes):
        env = NewMouselabEnv(gambles, attributes, reward, cost, ground_truth_dist = testreward, alpha=alpha, sample_term_reward = sample_term_reward)
    #for env in env_array:

        exp_return = 0
        actions_taken = []
        
        while True:
            possible_actions = list(env.actions())

            #take action that maximises estimated VOC
            action_taken = max(possible_actions, key = voc_estimate)

            _, rew, done, _=env._step(action_taken)
            
            exp_return+=rew
            actions_taken.append(action_taken)
            print(action_taken)
            if done:
                unopened += len(possible_actions) - 1 
                print(env.ground_truth, env.dist)
                #env._reset()
                break
        
        cumreturn += exp_return
        rewardlist.append(exp_return)
        #print(exp_return)
    
    avgclicks = (gambles + 1)*attributes - unopened/num_episodes
    print(avgclicks)
    print(cumreturn/num_episodes)
    return -cumreturn/num_episodes


# In[52]:


#observe_strategy()


# In[ ]:




