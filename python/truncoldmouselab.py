import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.stats import norm, truncnorm
import scipy.integrate as integrate
import gym
import random
import itertools as it
from agents import Agent
from evaluation import *
from distributions import cmax, smax, sample, expectation, Normal, PointMass, SampleDist, Normal, Categorical

ZERO = PointMass(0)

class TruncOldMouselabEnv(gym.Env):
    """MetaMDP for the Mouselab task."""
    
    term_state = '__term_state__'
    def __init__(self, gambles=4, attributes=5, reward=None, cost=0,
                 ground_truth=None, initial_states=None, randomness=1):
        
        self.gambles = gambles # number of gambles
        
        # distribution and number of attributes
        if hasattr(attributes, '__len__'):
            self.outcomes = len(attributes)
            self.dist = np.array(attributes)/np.sum(attributes)
        else:
            self.outcomes = attributes
            self.dist = np.random.dirichlet(np.ones(attributes)*randomness,size=1)[0]
        # reward for the payoffs
        self.reward = reward if reward is not None else Normal(1, 1)
        
        if hasattr(reward, 'sample'):
            self.iid_rewards = True
        else:
            self.iid_rewards = False
            
        self.cost = - abs(cost)
        self.initial_states = initial_states
        self.max = cmax
        self.init = np.array([self.reward,] * (self.gambles*self.outcomes))
        
        if ground_truth is False:
            self.ground_truth = ()
        elif ground_truth is not None:
            self.ground_truth = np.array(ground_truth)
        else:
            self.ground_truth = np.array(list(map(sample, self.init)))
            
        self.sample_term_reward = False
        self.term_action = self.gambles*self.outcomes
        self.reset()
#         self.ground_truth = np.array(ground_truth) if ground_truth is not None else None
#         self.grid = np.arange(self.gambles*self.outcomes).reshape((self.gambles, self.outcomes))
#         self.exact = hasattr(reward, 'vals')
#         if self.exact:
#             assert self.iid_rewards
#             self.max = cmax
#             self.init = np.array([self.reward,] * (self.gambles*self.outcomes))
#         else:
#             # Distributions represented as samples.
#             self.max = smax
#             self.init = np.array([self.reward.to_sampledist(),] * (self.gambles*self.outcomes))

    def _reset(self):
        if self.initial_states:
            self.init = random.choice(self.initial_states)
        self._state = self.init
        grid = np.array(self._state).reshape(self.gambles,self.outcomes)
        self.mus = [expectation(np.sum(self.dist*grid[g])) for g in range(self.gambles)]
        # todo: include max_mu
        # tmp: Works only for Normal
        self.vars = np.sum(self.dist**2*self.reward.sigma**2)*np.ones(self.gambles)
        self.lowers = self.reward.lower*np.ones(self.gambles)
        self.uppers = self.reward.upper*np.ones(self.gambles)
        return self.features(self._state)

    def _step(self, action):
#         print('step ' + str(action))
        if self._state is self.term_state:
            assert 0, 'state is terminal'
            # return None, 0, True, {}
        if action == self.term_action:
            # self._state = self.term_state
            if self.sample_term_reward:
                if self.ground_truth is not None:
                    gamble = self.best_gamble()
                    reward = self.ground_truth[gamble].sum()
                else:
                    reward = self.term_reward().sample()
            else:
                reward = self.expected_term_reward()
            self.last_state = self._state
            self._state = self.term_state
            done = True
        elif not hasattr(self._state[action], 'sample'):  # already observed
#             assert 0, self._state[action]
            reward = 0
            done = False
        else:  # observe a new node
            self._state = self._observe(action)
            reward = self.cost
            done = False
        return self.features(self._state), reward, done, {}

    def _observe(self, action):
#         print('obs ' + str(action))
        if self.ground_truth is not None:
            result = self.ground_truth[action]
        else:
            result = self._state[action].sample()
        s = list(self._state)
        gamble = action // self.outcomes
        option = action % self.outcomes
        self.mus[gamble] += self.dist[option]*(result - self.reward.expectation())
        self.vars[gamble] = max(0,self.vars[gamble] - self.dist[option]**2*self.reward.sigma**2)
        self.lowers[gamble] += result - self.dist[option]*self.reward.lower
        self.uppers[gamble] += result - self.dist[option]*self.reward.upper
        s[action] = result
        return tuple(s)

    def actions(self, state=None):
        """Yields actions that can be taken in the given state.

        Actions include observing the value of each unobserved node and terminating.
        """
        state = state if state is not None else self._state
        if state is self.term_state:
            return
        for i, v in enumerate(state):
            if hasattr(v, 'sample'):
                yield i
        yield self.term_action

    def results(self, state, action):
        """Returns a list of possible results of taking action in state.

        Each outcome is (probability, next_state, reward).
        """
        if action == self.term_action:
            # R = self.term_reward()
            # S1 = Categorical([self.term_state])
            # return cross(S1, R)
            yield (1, self.term_state, self.expected_term_reward(state))
        else:
            for r, p in state[action]:
                s1 = list(state)
                s1[action] = r
                yield (p, tuple(s1), self.cost)

    def features(self, state=None):
        state = state if state is not None else self._state
        return state


    def action_features(self, action, state=None):
        state = state if state is not None else self._state
        assert state is not None


        if action == self.term_action:
            return np.array([
                0,
                0,
                0,
                0,
                self.expected_term_reward(state)
            ])

        return np.array([
            self.cost,
            self.myopic_voi(action, state),
            self.vpi_action(action, state),
            self.vpi2(state),
            self.expected_term_reward(state)
        ])
    
    def gsample(self,gamble):
        if hasattr(gamble, 'sample'):
            return gamble.sample()
        return gamble
    
    # def vpi(self,state=None):
    #     gambles = [Normal(self.mus[i],np.sqrt(self.vars[i])) for i in range(self.gambles)]
    #     samples_max = np.amax([[self.gsample(gambles[i]) for i in range(self.gambles)] for _ in range(2500)],1)
    #     return np.mean(samples_max) - np.max(self.mus)
    
    def vpi2(self,state=None):
        state = state if state is not None else self._state
        grid = np.array(self._state).reshape(self.gambles,self.outcomes)
        gambles = grid.dot(self.dist)
        samples_max = np.amax([[self.gsample(gambles[i]) for i in range(self.gambles)] for _ in range(2500)],1)
        return np.mean(samples_max) - np.max(self.mus)
    
    def grid(self,state=None):
        if self._state == self.term_state:
            return np.array(self.last_state).reshape(self.gambles,self.outcomes)
        return np.array(self._state).reshape(self.gambles,self.outcomes)
   
    def vpi_action(self, action, state=None):
        #todo add action check
        state = state if state is not None else self._state
        #E[value if gamble corresponding to action is fully known]
        gamble = action//self.outcomes
        mus_wo_g = np.delete(self.mus,gamble)
        k = np.max(mus_wo_g)
        m = self.mus[gamble]
        s = np.sqrt(self.vars[gamble])
        a = self.lowers[gamble]
        b = self.uppers[gamble]
        e_higher = integrate.quad(lambda x: x*truncnorm.pdf(x,a,b,m,s), k, np.inf)[0]
        e_val = k*norm.cdf(k,m,s) + e_higher
        return e_val - np.max(self.mus)
    
    #todo edit
    def myopic_voi(self, action, state=None):
        #todo add action check
        state = state if state is not None else self._state
        #E[value if gamble corresponding to action is fully known]
#         print(action)
        gamble = action // self.outcomes
        outcome = action % self.outcomes
        mus_wo_g = np.delete(self.mus,gamble)
        k = np.max(mus_wo_g)
        m = self.mus[gamble]
        s = self.reward.sigma*self.dist[outcome]
        a = self.lowers[gamble]
        b = self.uppers[gamble]
        e_higher = integrate.quad(lambda x: x*truncnorm.pdf(x,a,b,m,s), k, np.inf)[0]
        e_val = k*norm.cdf(k,m,s) + e_higher
        return e_val - np.max(self.mus)

    
    def best_gamble(self, state=None):
        state = state if state is not None else self._state
        grid = np.array(state).reshape(self.gambles,self.outcomes)
        best_idx = np.argmax(self.mus)
#         best_gamble = max((self.dist*grid[g] for g in range(self.gambles)), default=ZERO, key=lambda x: sum(map(expectation,x)))
        return grid[best_idx]
    
    
    def term_reward(self, state=None):
        state = state if state is not None else self._state
        assert state is not None
        state_value = self.state_value(state)
        return state_value  
    
    def state_value(self, state=None):
        """A distribution over total rewards after the given node."""
        state = state if state is not None else self._state
        best_gamble = self.dist * self.best_gamble(state)
        return np.sum(best_gamble)   
    
    def expected_term_reward(self, state=None):
        sum_bg = self.term_reward(state)
        if hasattr(sum_bg, 'sample'):
            return sum_bg.expectation()
        return sum_bg