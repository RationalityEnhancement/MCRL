from IPython.display import clear_output
import itertools as it
import numpy as np
import json

from mouselab import MouselabEnv
from distributions import Normal
import skopt
from policies import LiederPolicy

def make_env(cost, ground_truth=False, initial_states=None):
    """Returns a MouselabEnv with branching [4,1,2].
    
    If `ground_truth` is True, the reward observed at a given node will be
    constant across runs on this env. This reduces variance of the return."""
    reward = Normal(0, 10).to_discrete(6)
    env = MouselabEnv([4,1,2], reward=reward, cost=cost, initial_states=initial_states)
    if ground_truth:
        env.ground_truth = np.array([0, *reward.sample(len(env.tree) - 1)])
    return env

def make_envs(cost, n=100, ground_truth=None, initial_states=None):
    # Note, ground_truth can be an int in which case it acts as a random seed.
    if ground_truth is not None:
        np.random.seed(ground_truth)
        return [make_env(cost, True, initial_states) for _ in range(n)]
    else:
        return [make_env(cost, False, initial_states)] * n

def filename(cost):
    c = round(float(cost), 5)
    return f'data/412_{c}.pkl'

def read_bo_result(cost):
    return skopt.load(filename(cost))

def read_bo_policy(cost):
    result = read_bo_result(cost)
    return LiederPolicy(result.specs['info']['theta'])

def read_state_actions():
    with open('data/state_actions.json') as f:
        data = json.load(f)
    result = {}
    for cost in data:
        result[float(cost)] = r = {}
        env = make_env(float(cost))
        def parse_state(state):
            return tuple(env.reward if x == '__' else float(x)
                  for x in state)
        def parse_action(action):
            return env.term_action if action == '__TERM_ACTION__' else action
        r['states'] = list(map(parse_state, data[cost]['states']))
        r['actions'] = list(map(parse_action, data[cost]['actions']))
    return result

state_actions = read_state_actions()

def join(*args, sep=' '):
    return sep.join(map(str, args))

def log_return(func):
    def wrapped(*args, **kwargs):
        r = func(*args, **kwargs)
        print('{} returns {}'.format(func.__name__, r))
        return r
    return wrapped

def logged(condition=lambda r: True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if condition(result):
                print(func.__name__, args, kwargs, '->', result)
            return result
        return wrapper
    return decorator

import time
def timed(method):
    def dec(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r %2.2f sec' % (method.__name__, te-ts))
        return result
    return dec



def cum_returns(rewards):
    return np.flip(np.cumsum(np.flip(rewards, 0)), 0)

class Labeler(object):
    """Assigns unique integer labels."""
    def __init__(self, init=()):
        self._labels = {}
        self._xs = []
        for x in init:
            self.label(x)

    def label(self, x):
        if x not in self._labels:
            self._labels[x] = len(self._labels)
            self._xs.append(x)
        return self._labels[x]

    def unlabel(self, label):
        return self._xs[label]

    __call__ = label

def clear_screen():
    print(chr(27) + "[2J")
    clear_output()

import time
def show_path(env, trace, render='human'):
    env.reset()
    env.render(mode=render)
    for a in trace['actions']:
        env.step(a)
        input('>')
        env.render(mode=render)


import heapq
class PriorityQueue(list):
    def __init__(self, key, max_first=True):
        self.key = key
        self.inv = -1 if max_first else 1

    def pop(self):
        return heapq.heappop(self)[1]
        
    def push(self, item):
        heapq.heappush(self, (self.inv * self.key(item), item))

def softmax(x):
    ex = np.exp(x)
    return ex / ex.sum()


    # def encode(x):
    #     s = 0
    #     for f, n in zip(x, fs):
    #         s *= n
    #         s += f
    #     return s
            
    # def decode(s):
    #     x = []
    #     for n in reversed(fs):
    #         x.append(s % n)
    #         s //= n
    #     return tuple(reversed(x))
        # super().__init__()

def dict_product(d):
    """All possible combinations of values in lists in `d`"""
    for k, v in d.items():
        if not isinstance(v, list):
            d[k] = [v]

    for v in list(it.product(*d.values())):
        yield dict(zip(d.keys(), v))