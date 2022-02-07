import numpy as np
from scipy.stats import norm, dirichlet
import scipy.integrate as integrate
import quadpy
import gym
import random
from functools import lru_cache
import matplotlib.pyplot as plt
import pandas as pd
import numbers
from contracts import contract

from distributions import *

class DistRV(object):
    """An object that returns samples from a dirichlet distribution"""
    def __init__(self, alpha, attributes, ground_truth = None):
        super().__init__()
        self.alpha = alpha
        self.attributes = attributes
        self.num_unobs = attributes
        self.init = np.ones(attributes)*-1
        self.state = self.init
        
        if ground_truth is False:
            self.ground_truth = False
        elif ground_truth is not None:
            self.ground_truth = np.array(ground_truth)
        else:
            self.ground_truth = self.sample_all()
        
    def __repr__(self):
        return 'DistRV(a=' + str(self.alpha) + '): [' + ", ".join(self.print_dist()) + "]" 

    def _reset(self):
        self.state = np.copy(self.init) 
        self.num_unobs = self.attributes
        return self.state
    
    def print_dist(self):
        return ['{:.3f}'.format(self.state[i]) if self.state[i] != -1 else 'p' +str(i) for i in range(self.attributes)]
    
    def observe_p(self, i):
        if self.state[i] == -1:
            self.num_unobs -= 1
            if self.ground_truth is not False:
                self.state[i] = self.ground_truth[i]
            else:
                self.state[i] = np.random.beta(self.alpha, self.alpha*(self.num_unobs - 1))
        return self.state
    
    def sample_p(self, i, n=1, expectation = False):
        p_vec = np.repeat(self.state[None,:],n,axis=0)
        if self.num_unobs == 0:
            return p_vec
        if self.state[i] == -1:
            if self.num_unobs == 1:
                p_vec[:,i] = (1-np.sum(p_vec[:,p_vec[0] != -1],1))
            else:
                p_vec[:,i] = np.random.beta(self.alpha, self.alpha*(self.num_unobs - 1),size=n)
        if expectation:
            filler = (1-np.sum(p_vec[:,p_vec[0] != -1],1)[:,None])
            if self.num_unobs > 1:
                filler /= (self.num_unobs - 1)
            p_vec[:,p_vec[0] == -1] = filler
        return np.squeeze(p_vec)
    
    def sample_all(self, n=1):
        p_vec = np.repeat(self.state[None,:],n,axis=0)
        if self.num_unobs == 0:
            return p_vec
        else:
            alpha_vec = np.ones(self.num_unobs)*self.alpha
            sampled_unobs = np.random.dirichlet(alpha_vec,size=n)
            p_vec[:,self.state == -1] = (1-np.sum(self.state[self.state != -1])) * sampled_unobs
        return np.squeeze(p_vec)
    
    def expectation(self):
        p_vec = np.copy(self.state)
        if self.num_unobs != 0:
            p_vec[p_vec == -1] = (1-np.sum(p_vec[p_vec != -1]))/self.num_unobs
        return p_vec


class NewMouselabEnv(gym.Env):
    """MetaMDP for the Mouselab task."""

    term_state = '__term_state__'
    def __init__(self, gambles=4, attributes=5, reward = None, cost=0,
                 ground_truth=None, ground_truth_dist = None, alpha=1, sample_term_reward=False, quantization = False):

        self.gambles = gambles # number of gambles
        self.quantization = quantization
        self.attributes = attributes
        self.outcomes = attributes

        self.distRV = DistRV(alpha, attributes, ground_truth = ground_truth)
        self.reward = reward

        #extra parameter: if the agent doesn't know the actual distribution: 
        #ground_truth sampled from self.ground_truth_dist but features computed using self.reward
        self.ground_truth_dist = ground_truth_dist if ground_truth_dist is not None else reward
        
        # TODO: check correctness of the code when quantization is set to true
        if quantization:
            self.discrete_reward = self.ground_truth_dist.to_discrete(quantization)

        if hasattr(reward, 'sample'):
            self.iid_rewards = True
        else:
            self.iid_rewards = False

        self.cost = - abs(cost)
        self.max = cmax
        self.init_rewards = tuple([self.reward,] * (self.gambles*self.outcomes))
        self.init = (self.distRV, self.init_rewards)

        self.ground_truth_init = tuple([self.ground_truth_dist,] * (self.gambles*self.outcomes))
        # self.ground_truth only includes rewards
        # self.distRV.ground_truth has the distribution ground truth
        if ground_truth is False:
            self.ground_truth = False
        elif ground_truth is not None:
            self.ground_truth = np.array(ground_truth)
        else:
            if self.quantization:
                self.ground_truth = np.array([self.discrete_reward.sample() for _ in self.init])
            else:
                self.ground_truth = np.array(list(map(sample, self.ground_truth_init)))

        self.sample_term_reward = sample_term_reward
        self.term_action = (self.gambles+1)*self.outcomes
        self.reset()

    def _reset(self):
        self.init[0]._reset()
        self.vpi.cache_clear()
        self.vpi_action.cache_clear()
        self._state = self.init
        self.dist = self.distRV.expectation()
        gambles = self.gamble_dists()
        self.mus = [expectation(g) for g in gambles]
        self.gdists = gambles
        return self._state

    def _step(self, action):
        self.vpi.cache_clear()
        self.vpi_action.cache_clear()
        if self._state is self.term_state:
            assert 0, 'state is terminal'
            # return None, 0, True, {}
        if action >= self.term_action:
            if self.sample_term_reward:
                if self.ground_truth is not False:
                    best_idx = np.argmax(self.mus)
                    gt_grid = self.ground_truth.reshape(self.gambles,self.outcomes)
                    reward = self.distRV.ground_truth.dot(gt_grid[best_idx])
                else:
                    reward = sample(self.term_reward())
            else:
                reward = self.expected_term_reward()
            self.last_state = self._state
            self._state = self.term_state
            done = True
        elif self.term_action > action >= self.attributes:
            if not hasattr(self._state[1][action-self.attributes], 'sample'):  # already observed reward
                reward = 0      
            else:  # observe a new node
                self._state = self._observe(action)
                reward = self.cost
                gamble = action // self.outcomes
                gamble -=1
                grid = np.array(self._state[1]).reshape(self.gambles, self.outcomes)
                gdist = np.dot(grid[gamble], self.dist)
                self.mus[gamble] = expectation(gdist)
                self.gdists[gamble] = gdist
            done = False
        else:
            if not self._state[0].state[action] == -1: # already observed attribute
                reward = 0
            else:  # observe a new attribute
                self._state = self._observe(action)
                reward = self.cost #todo: possibly have a separate cost for p observations
                gambles = self.gamble_dists()
                self.mus = [expectation(g) for g in gambles]
                self.gdists = gambles
            done = False
        return self._state, reward, done, {}

    def _observe(self, action):
        if action >= self.attributes:
            action -= self.attributes
            if self.ground_truth is not False:
                result = self.ground_truth[action]
            elif self.quantization:
                assert hasattr(self._state[action], 'sample')
                result = self.discrete_reward.sample()
            else:
                result = self._state[action].sample()
            s = list(self._state[1])
            
            s[action] = result
            return (self._state[0],tuple(s))
        else:
            # edit so it is a temporary change unless assigned
            self._state[0].observe_p(action)
            self.dist = self._state[0].expectation()
            return self._state

    def actions(self, state=None):
        """Yields actions that can be taken in the given state.

        Actions include observing the value of each unobserved node and terminating.
        """
        probs = state[0] if state is not None else self._state[0]
        rewards = state[1] if state is not None else self._state[1]
        if state is None:
            if self._state is self.term_state:
                return
        for i in range(self.attributes):
            if probs.state[i] == -1:
                yield i
        for i, v in enumerate(rewards):
            if hasattr(v, 'sample'):
                yield i + self.attributes
        yield self.term_action

    def action_features(self, action):
        if action == self.term_action:
            return np.array([
                0,
                0,
                0,
                0,
                self.expected_term_reward()
            ])
        else:
            gamble = action // self.outcomes
            gamble = -action if gamble == 0 else gamble
            return np.array([
                self.cost,
                self.myopic_voi(action),
                self.vpi_action(gamble),
                self.vpi(),
                self.expected_term_reward()
            ])

    def gamble_dists(self, state = None, sample_all = False):
        state = state if state is not None else self._state
        sdist = state[0].sample_all() if sample_all else self.dist
        grid = np.array(state[1]).reshape(self.gambles, self.outcomes)
        return np.dot(grid, sdist)
    
    def print_state(self,state=None):
        state = state if state is not None else self._state
        if state is self.term_state:
            return self.print_state(state = self.last_state)
        return pd.DataFrame(self.grid(state),columns=state[0].print_dist())
    
    def grid(self,state=None):
        state = state if state is not None else self._state
        return np.array(state[1]).reshape(self.gambles,self.outcomes)

    @lru_cache(None)
    def vpi(self):
        sdist = self._state[0].sample_all(1000)
        grid = np.array(self._state[1]).reshape(self.gambles,self.outcomes)
        rgrid = np.repeat(grid[None,:,:],1000,axis=0)
        rgrid = np.vectorize(lambda g: sample(g), otypes = [float])(rgrid)
        sampled_gambles = np.einsum('ijk,ik->ij',rgrid,sdist)
        e_val = np.mean(np.amax(sampled_gambles,1))

        return e_val - np.max(self.mus)

    @lru_cache(None)
    def vpi_action(self, gamble):
        #E[value if gamble corresponding to action is fully known]
        if gamble > 0:
            gamble -= 1
            mus_wo_g = np.delete(self.mus,gamble)
            k = np.max(mus_wo_g)
            if hasattr(self.gdists[gamble], 'sample'):
                e_higher = self.gdists[gamble].integrate_xfx(low = k)
                e_val = k*self.gdists[gamble].cdf(k) + e_higher
            
            else:
                e_val = max(self.gdists[gamble], k)
        else:
            action = -1*gamble
            sdist = self._state[0].sample_all(1000)
            grid = np.array(self._state[1]).reshape(self.gambles,self.outcomes)
            rgrid = np.repeat(grid[None,:,:],1000,axis=0)
            rgrid[:,:,action] = np.vectorize(lambda g: sample(g), otypes = [float])(rgrid[:,:,action])
            rgrid = np.vectorize(lambda g: expectation(g), otypes = [float])(rgrid)
            sampled_gambles = np.einsum('ijk,ik->ij',rgrid,sdist)
            e_val = np.mean(np.amax(sampled_gambles,1))
        return e_val - np.max(self.mus)

    #@contract
    def myopic_voi(self, action):
        #E[value if gamble corresponding to action is fully known]
        if action >= self.attributes:
            action -= self.attributes
            gamble = action // self.outcomes
            outcome = action % self.outcomes
            mus_wo_g = np.delete(self.mus,gamble)
            k = np.max(mus_wo_g)

            grid = np.array(self._state[1]).reshape(self.gambles,self.outcomes)
            exp_gamble = []
            for g in grid[gamble]:
                exp_gamble.append(expectation(g))

            exp_gamble[outcome] = grid[gamble][outcome]    

            gambledist = np.dot(exp_gamble,self.dist)
            if hasattr(gambledist, 'sample'):
                e_higher = gambledist.integrate_xfx(low = k)
                e_val = k*gambledist.cdf(k) + e_higher
            else:
                e_val = max(gambledist, k)
            
        else:
            egrid = (np.vectorize(lambda g: expectation(g), otypes = [float]) 
                (np.array(self._state[1]))).reshape(self.gambles,self.outcomes)
            sdist = self._state[0].sample_p(action, n = 1000, expectation = True)
            smus = sdist.dot(egrid.T)
            e_val = np.mean(np.amax(smus,1))
        return e_val - np.max(self.mus)

    def term_reward(self, state=None):
        state = state if state is not None else self._state
        grid = np.array(state[1]).reshape(self.gambles,self.outcomes)
        best_idx = np.argmax(self.mus)
        return self.dist.dot(grid[best_idx])

    def expected_term_reward(self):
        return np.max(self.mus)

    def select_gamble(self, gamble):
        if self.sample_term_reward:
            gt_grid = self.ground_truth.reshape(self.gambles,self.outcomes)
            return self.distRV.ground_truth.dot(gt_grid[gamble])
        else:
            return self.mus[gamble]
