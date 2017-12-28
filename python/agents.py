"""Agents that operate in discrete fully observable environments."""

from collections import namedtuple, defaultdict, Counter, deque
import itertools as it
import numpy as np
from abc import ABC, abstractmethod
from utils import clear_screen, PriorityQueue
import time
np.set_printoptions(precision=3, linewidth=200)

from tqdm import tqdm, trange, tnrange
from copy import deepcopy
from toolz.curried import *

# from policies import *
# from value_functions import *
# ========================== #
# ========= Agents ========= #
# ========================== #

class RegistrationError(Exception): pass

class Agent(ABC):
    """An agent that can run openai gym environments."""
    def __init__(self):
        self.env = None
        self.policy = None
        self.ep_trace = None
        self.value_functions = []
        self.i_episode = 0
        self.memory = None

    def register(self, obj):
        """Attaches a component or env to this agent."""
        if hasattr(obj, 'step'):  # gym Env
            self.env = obj
        elif hasattr(obj, 'act'):
            self.policy = obj
            obj.attach(self)
        elif hasattr(obj, 'predict'):
            self.value_functions.append(obj)
            obj.attach(self)
        elif hasattr(obj, 'batch'):
            self.memory = obj
        else:
            raise ValueError('Cannot register {}'.format(obj))

    def run_episode(self, render=False, max_steps=1000, interact=False,
                    verbose=False, reset=True):
        """Runs a single episode, returns a complete trace of the episode."""
        if not self.env:
            raise RegistrationError('No environment registered.')
        if not self.policy:
            raise RegistrationError('No policy registered.')

        self.log = print if verbose else (lambda *args: None)

        trace = self.ep_trace = defaultdict(list)
        trace.update({
                    'i_episode': self.i_episode,
                    'states': [],
                    'actions': [],
                    'rewards': [],
                    'finished': False,
                    'return': None
                })
        if reset:
            new_state = self.env.reset()
        else:
            new_state = self.env._state
        # trace['_state'] = self.env._state
        self._start_episode(new_state)
        done = False
        for i_step in range(max_steps):
            state = new_state

            self._render(render)
            action = self.policy.act(state)
            new_state, reward, done, info = self.env.step(action)
            self._experience(state, action, new_state, reward, done)
            
            trace['states'].append(state)
            trace['actions'].append(action)
            trace['rewards'].append(reward)

            if done:
                trace['finished'] = True
                self._render(render)
                break


        trace['states'].append(new_state)  # final state
        trace['return'] = sum(trace['rewards'])
        if self.memory is not None:
            self.memory.add(trace)
        self._finish_episode(trace)
        self.i_episode += 1
        return dict(trace)

    def run_many(self, n_episodes, pbar=True, track=(), **kwargs):
        """Runs several episodes, returns a summary of results."""
        data = defaultdict(list)
        range_ = tnrange if pbar else range
        for _ in range_(n_episodes):
            trace = self.run_episode(**kwargs)
            data['n_steps'].append(len(trace.pop('states')))
            # data['i_episode'].append(trace.pop('i_episode'))
            # data['return'].append(trace.pop('return'))
            # data['finished'].append(trace.pop('finished'))
            trace.pop('actions')
            trace.pop('rewards')
            for k, v in trace.items():
                data[k].append(v)

        return dict(data)

    def _start_episode(self, state):
        self.policy.start_episode(state)
        for vf in self.value_functions:
            vf.start_episode(state)

    def _finish_episode(self, trace):
        self.policy.finish_episode(trace)
        for vf in self.value_functions:
            vf.finish_episode(trace)
        

    def _experience(self, s0, a, s1, r, done):
        self.policy.experience(s0, a, s1, r, done)
        for vf in self.value_functions:
            vf.experience(s0, a, s1, r, done)

    def _render(self, mode):
        if mode == 'step':
            x = input('> ')
            while x:
                print(eval(x))
                x = input('> ')
            clear_screen()
            self.env.render()
        elif mode == 'clear':
            clear_screen()
            self.env.render()
        elif mode == 'auto':
            time.sleep(.4)
            clear_screen()
            self.env.render()
        elif mode:
            self.env.render(mode=mode)


class Component(ABC):
    """A very abstract base class."""
    def __init__(self):
        super().__init__()
        self.agent = None
        self.saved = defaultdict(list)

    def experience(self, state, action, new_state, reward, done):
        """Learn from the results of taking action in state.

        state: state in which action was taken.
        action: action taken.
        new_state: the state reached after taking action.
        reward: reward received after taking action.
        done: True if the episode is complete.
        """
        pass

    def start_episode(self, state):
        """This function is run when an episode begins, starting at state."""
        pass

    def finish_episode(self, trace):
        """This function is run when an episode ends."""
        return

    def attach(self, agent):
        self.agent = agent

    @property
    def env(self):
        return self.agent.env

    @property
    def i_episode(self):
        return self.agent.i_episode

    @property
    def observation_shape(self):
        return self.env.observation_space.shape

    @property
    def state_size(self):
        s = self.env.observation_space.shape
        assert len(s) == 1
        return s[0]

    @property
    def n_action(self):
        return self.env.action_space.n
    
    @property
    def memory(self):
        return self.agent.memory

    @property
    def ep_trace(self):
        return self.agent.ep_trace

    def log(self, *args):
        self.agent.log(*args)

    def save(self, key, val):
        self.saved[key].append(val)


class Memory(object):
    """Remembers past experiences."""
    def __init__(self, size=100000):
        self.states = deque(maxlen=size)
        self.actions = deque(maxlen=size)
        self.rewards = deque(maxlen=size)
        self.returns = deque(maxlen=size)

    def add(self, trace):
        self.states.extend(trace['states'])
        self.actions.extend(trace['actions'])
        self.actions.append(None)
        self.rewards.extend(trace['rewards'])
        self.rewards.append(0)
        self.returns.extend(np.flip(np.cumsum(np.flip(trace['rewards'], 0)), 0))
        self.returns.append(0)



        # self.experiences.extend(zip(trace['states'][:-1],
        #                             trace['actions'],
        #                             # trace['states'][1:],
        #                             trace['rewards'],
        #                             np.flip(np.cumsum(np.flip(trace['rewards'], 0)), 0)))
        # self.deque.append({'states': states, 'rewards': rewards, 'returns': returns})

    # def episodes(self, size, n=1):
    #     size = min(size, len(self.deque))
    #     if not self.deque:
    #         return
    #     for _ in range(n):
    #         yield np.random.choice(self.deque, size, replace=False)

    def batch(self, size):
        size = min(size, len(self.states))
        idx = np.random.choice(len(self.states), size=size, replace=False)
        return idx
        # return (self.experiences[i] for i in idx)


# class Memory(object):
#     """Remembers past experiences."""
#     Memory = namedtuple('Memory', ['states', 'rewards', 'returns'])
#     def __init__(self, size=100000):
#         self.episodes = deque(maxlen=size)
#         self.experiences
#         self.size = size

#     def add(self, trace):
#         # TODO this wastes RAM
#         states = np.stack(trace['states'])
#         returns = np.flip(np.cumsum(np.flip(trace['rewards'], 0)), 0)
#         rewards = np.array(trace['rewards'])
#         self.deque.append({'states': states, 'rewards': rewards, 'returns': returns})

#     def episodes(self, size, n=1):
#         size = min(size, len(self.deque))
#         if not self.deque:
#             return
#         for _ in range(n):
#             yield np.random.choice(self.deque, size, replace=False)


def run_episode(policy, env):
    agent = Agent()
    agent.register(env)
    agent.register(policy)
    return agent.run_episode()

def interactions(x):
    return [a * b for a, b in it.combinations(x, 2)]



class Model(object):
    """Simulated environment"""
    def __init__(self, env):
        self.env = deepcopy(env)
      
    def options(self, state):
        for a in range(self.env.action_space.n):
            self.env._state = state
            obs, r, done, info = self.env.step(a)
            yield a, self.env._state, r, done




