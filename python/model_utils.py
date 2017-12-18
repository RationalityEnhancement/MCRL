from distributions import Normal
from mouselab import MouselabEnv
import numpy as np
import json
import skopt
from policies import LiederPolicy
from agents import Agent

def clear_screen():
    print(chr(27) + "[2J")
    clear_output()

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

def make_env(cost, ground_truth=False, **kwargs):
    """Returns a MouselabEnv with branching [4,1,2].
    
    If `ground_truth` is True, the reward observed at a given node will be
    constant across runs on this env. This reduces variance of the return."""
    reward = Normal(0, 10).to_discrete(6)
    env = MouselabEnv([4,1,2], reward=reward, cost=cost, **kwargs)
    if hasattr(ground_truth, 'len'):
        env.ground_truth = np.array(ground_truth)
    elif ground_truth:
        env.ground_truth = np.array([0, *reward.sample(len(env.tree) - 1)])
    return env

def make_envs(cost, n=100, ground_truth=None, **kwargs):
    # Note, ground_truth can be an int in which case it acts as a random seed.
    if ground_truth is not None:
        np.random.seed(ground_truth)
        return [make_env(cost, True, **kwargs) for _ in range(n)]
    else:
        return [make_env(cost, False, **kwargs)] * n

def filename(cost, note=''):
    c = round(float(cost), 5)
    if note:
        note += '_'
    return f'data/policy_{note}{c}.pkl'

def read_bo_result(cost, note=''):
    return skopt.load(filename(cost, note))

def read_bo_policy(cost, note=''):
    result = read_bo_result(cost, note)
    return LiederPolicy(result.specs['info']['theta'])

ENV = make_env(0)

def parse_state(state):
    return tuple(ENV.reward if x == '__' else float(x)
                 for x in state)
def parse_action(action):
    return ENV.term_action if action == '__TERM_ACTION__' else action

def read_state_actions(cost):
    with open(f'data/human_state_actions_{cost:.2f}.json') as f:
        data = json.load(f)


    return {'states': list(map(parse_state, data['states'])), 
            'actions': list(map(parse_action, data['actions']))}
