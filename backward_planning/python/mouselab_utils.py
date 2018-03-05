import numpy as np
from mouselab import MouselabEnv
from distributions import Normal
from toolz import get, partition_all
import joblib

ENV_TYPES = ["constant_high", "constant_low", "increasing", "decreasing"]

def make_envs(cost=1.00, n=100, seed=None, variance_structure="constant_high"):
    if seed is not None:
        np.random.seed(seed)

    sigmas = get(variance_structure, {
        "constant_high": [0,20,20,20],
        "increasing": [0, 2, 4, 20],
        "decreasing": [0,20,10,5],
        "constant_low": [0,1,1,1],
    })
    
    def reward(depth):
        if depth > 0:
            return Normal(0, sigmas[depth]).to_discrete(6)
        return 0.
    
    branching = [4,1,2]
    envs = [MouselabEnv.new_symmetric(branching, reward, cost=cost)
            for _ in range(n)]
    
    return envs


def make_env(env_type, cost=1., seed=None, **kwargs):
    if seed is not None:
        np.random.seed(seed)

    sigmas = get(env_type, {
        "constant_high": [0,20,20,20],
        "increasing": [0, 2, 4, 20],
        "decreasing": [0,20,10,5],
        "constant_low": [0,1,1,1],
    })
    
    def reward(depth):
        if depth > 0:
            return Normal(0, sigmas[depth]).to_discrete(6)
        return 0.
    
    branching = [4,1,2]
    return MouselabEnv.new_symmetric(branching, reward, cost=cost, **kwargs)
    


def encode_state(state):
    return ' '.join('_' if hasattr(x, 'sample') else str(x)
                    for x in state)

    
from agents import Agent
import pandas as pd

def evaluate(policy, envs):
    agent = Agent()
    def run_env(policy, env):
        agent.register(env)
        agent.register(policy)
        tr = agent.run_episode()
        return {'util': tr['return'],
                'observations': len(tr['actions']) - 1}
    return pd.DataFrame(run_env(policy, env) for env in envs)

__ENVS = None
__AGENT = Agent()
__CHUNKS = None

def eval_chunk(i, return_mean=True):
    # Each process should start with a different random seed.
    np.random.seed(np.random.randint(1000) + i)
    returns = []
    for env in __CHUNKS[i]:
        __AGENT.register(env)
        returns.append(__AGENT.run_episode()['return'])
    if return_mean:
        return np.mean(returns)
    else:
        return returns

def get_util(policy, envs, parallel=None, return_mean=True):
    if parallel is None:
        util = evaluate(policy, envs).util
        if return_mean:
            return util.mean()
        else:
            return util
    else:
        np.random.randint(1000) # cycle the random number generator
        global __CHUNKS
        chunk_size = len(envs) // parallel.n_jobs
        __CHUNKS = list(partition_all(chunk_size, envs))
        __AGENT.register(policy)
        jobs = (joblib.delayed(eval_chunk)(i, return_mean)
                for i in range(len(__CHUNKS)))
        result = parallel(jobs)
        if return_mean:
            return np.mean(result)
        else:
            return np.concatenate(result)
