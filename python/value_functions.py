from collections import namedtuple, defaultdict, Counter, deque
import itertools as it
import numpy as np
from abc import ABC, abstractmethod
from utils import clear_screen, PriorityQueue
import time
np.set_printoptions(precision=3, linewidth=200)

from tqdm import tqdm, trange, tnrange
from copy import deepcopy
# from keras.utils import to_categorical

from agents import Component


class ValueFunction(Component):
    """Learns values."""
    def __init__(self, learn_rate=.1, discount=1):
        super().__init__()
        self.discount = discount
        self.learn_rate = learn_rate

    def features(self, s):
        if hasattr(self.env, 'nS'):
            x = [0] * self.env.nS
            x[s] = 1
            return x
        else:
            return np.r_[1, s]

    def predict(self, s):
        return 0
     

class ActionValueFunction(ValueFunction):
    """Values of state action pairs."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, s):
        x = self.features(s)
        return self.model.predict(x)


class QFunction(object):
    """Q(s, a)"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def predict(s, a):
        ...



      



class NeuralQ(ActionValueFunction):
    """Neural network approximation of Q function."""
    def __init__(self, arg):
        super().__init__()
        self.model = self._build_model()

    def _build_model(self):
        actor = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu',
                  kernel_initializer='he_uniform'),
            # Dense(24, activation='relu',
            #       kernel_initializer='he_uniform'),
            Dense(self.n_action, activation='softmax',
                  kernel_initializer='he_uniform')
        ])
        # actor.summary()
        actor.compile(loss='mse',
                      optimizer=Nadam(self.learn_rate))
        return actor
      

class LinearQ(ActionValueFunction):
    """Learns a linear Q function by SGD."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None

    def attach(self, agent):
        super().attach(agent)
        sx = len(self.features(self.env.reset()))
        sy = self.agent.n_actions
        shape = (sx, sy)
        self.model = LinearSGD(shape, learn_rate=self.learn_rate)
    
    def experience(self, s0, a, s1, r, done):
        x0, x1 = self.features(s0), self.features(s1)
        target = self.model.predict(x1)
        target[a] = r + self.discount * np.max(self.model.predict(x1))
        self.model.update(x0, target)

    def predict(self, s):
        x = self.features(s)
        return self.model.predict(x)

# class LinearQ(ActionValueFunction):
#     """Learns a linear Q function by SGD."""
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.model = None

#     def attach(self, agent):
#         super().attach(agent)
#         sx = len(self.features(self.env.reset()))
#         sy = self.agent.n_actions
#         shape = (sx, sy)
#         self.model = LinearSGD(shape, learn_rate=self.learn_rate)
    
#     def experience(self, s0, a, s1, r, done):
#         x0, x1 = self.features(s0), self.features(s1)
#         target = self.model.predict(x1)
#         target[a] = r + self.discount * np.max(self.model.predict(x1))
#         self.model.update(x0, target)

#     def predict(self, s):
#         x = self.features(s)
#         return self.model.predict(x)


class StateValueFunction(ValueFunction):
    """Learns a linear V function by SGD."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


from models import BayesianRegression
class BayesianRegressionV(StateValueFunction):
    """Learns a linear V function by SGD."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None

    def attach(self, agent):
        super().attach(agent)
        self.model = BayesianRegression(np.zeros(self.state_size), sigma_w=10)

    def predict(self, state, return_var=False):
        return self.model.predict(state, return_var=return_var)

    # def finish_episode(self, trace):
    #     for batch in self.memory.batch(10):
    #         states, rewards, returns = batch

    #     self.model.update(batch['states'][:-1], batch['returns'])  # don't learn about final state
    #     self.save('w', self.model.w.copy())
    #     self.save('sigma_w', self.model.sigma_w.copy())


# from models import BayesQ, BayesianRegression
class BayesianRegressionQ(StateValueFunction):
    """Learns a linear Q function by Bayesian regression."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None

    def attach(self, agent):
        super().attach(agent)
        # self.model = BayesQ(np.zeros((self.state_size, self.n_action)), sigma_w=10.)


    def predict(self, state, return_var=False):
        return 
        # return self.model.predict(state, return_var=return_var)

    def finish_episode(self, trace):
        idx = self.memory.batch(1000)

        states = []
        actions = []
        qs = []
        for i in idx:
            if self.memory.actions[i] is None:
                continue  # can't update for final state
            states.append(self.memory.states[i])
            actions.append(self.memory.actions[i])
            # value = self.model.predict([self.memory.states[i+1]]).max()
            value = self.memory.returns[i+1]
            qs.append(self.memory.rewards[i] + value)
            
        # actions = to_categorical(actions, num_classes=self.n_action)
        # self.model.update(np.stack(states), actions, np.array(qs), 1)



        # exps = self.memory.batch(1000)
        # states, actions, next_states, returns, rewards = zip(*exps)
        # target = self.model.predict(states)
        


        # for ep in self.memory.episodes(10):
        #     print(batch)
        #     self.model.update(batch['states']
        #                       batch['returns'])

        self.save('w', self.model.w.copy())
        self.save('sigma_w', self.model.sigma_w.copy())


class TDLambdaV(StateValueFunction):
    """Learns a linear value function with TD lambda."""
    def __init__(self, trace_decay=0, **kwargs):
        super().__init__(**kwargs)
        self.trace_decay = trace_decay
        self.trace = None

    def attach(self, agent):
        self.agent = agent
        shape = len(self.features(self.env.reset()))
        self.trace = np.zeros(shape)
        self.theta = np.zeros(shape)
        self.theta_update = np.zeros(shape)

    def start_episode(self, state):
        self.theta = self.theta_update.copy()

    def experience(self, s0, a, s1, r, done):
        target = r + self.discount * self.predict(s1)
        x = self.features(s0)
        prediction = x @ self.theta
        error = target - prediction
        self.trace = self.trace_decay * self.trace + x
        self.theta_update += self.learn_rate * error * self.trace
        # self.theta_update *= self.decay

    def predict(self, s):
        x = self.features(s)
        return x @ self.theta

    def finish_episode(self, trace):
        self.ep_trace['theta_v'] = self.theta.copy()


class FixedV(StateValueFunction):
    """User-specified value function."""
    def __init__(self, theta):
        super().__init__()
        self.theta = np.array(theta)

    def predict(self, s):
        x = self.features(s)
        return x @ self.theta

    def finish_episode(self, trace):
        self.ep_trace['theta_v'] = self.theta.copy()


class FixedQ(StateValueFunction):
    """User-specified action value function."""
    def __init__(self, theta):
        super().__init__()
        self.theta = np.array(theta)

    def predict(self, s):
        
        x = self.features(s)
        return x @ self.theta

    def finish_episode(self, trace):
        self.ep_trace['theta_v'] = self.theta.copy()


class LinearV(StateValueFunction):
    """Learns a linear V function by SGD."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None

    def attach(self, agent):
        super().attach(agent)
        sx = len(self.features(self.env.reset()))
        sy = 1
        shape = (sx, sy)
        self.model = LinearSGD(shape, learn_rate=self.learn_rate)

    def experience(self, s0, a, s1, r, done):
        x = self.features(s0)
        # TODO: should we set target to 0 if done is True?
        target = r + self.discount * self.predict(s1)
        self.model.update(x, target)

    def predict(self, s):
        x = self.features(s)
        return self.model.predict(x)[0]
    
    def finish_episode(self, trace):
        self.ep_trace['theta_v'] = self.model.theta[:, 0].copy()


# class MemV(StateValueFunction):
#     """docstring for MemV"""
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.tbl = defaultdict(list)

#     def finish_episode(self, trace):
#         returns = concatv(reversed(np.cumsum(list(reversed(trace['rewards'])))), [0])
#         for s, r in zip(trace['states'], returns):
#             self.tbl[s].append(r)



# from sklearn.linear_model import SGDRegressor
# class MonteCarloV(ValueFunction):
#     """Learns a linear value function with every-step Monte Carlo."""
#     def __init__(self, env, **kwargs):
#         super().__init__(env, **kwargs)
#         self.model = SGDRegressor()

#     def finish_episode(self, trace):
#         X = np.array([self.features(s) for s in trace['states']])
#         y = list(reversed(np.cumsum(list(reversed(trace['rewards'])))))
#         y.append(0)  # value of final state
#         self.model.partial_fit(X, y)

#     def predict(self, s):
#         if self.model.coef_ is not None:
#             return self.model.predict(self.features(s).reshape(1, -1))[0]
#         else:
#             return 0

#     @property
#     def theta(self):
#         return self.model.coef_



