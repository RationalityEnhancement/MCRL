from distributions import Normal
from mouselab import MouselabEnv
import numpy as np
import json
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