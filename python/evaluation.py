from skopt import gp_minimize
import pandas as pd
import numpy as np

from agents import Agent
from policies import LiederPolicy, FixedPlanPolicy
from contexttimer import Timer

__ENVS = None
__AGENT = Agent()

def eval_one(i): 
    __AGENT.register(__ENVS[i])
    return __AGENT.run_episode()['return']

def eval_chunk(i): 
    
    __AGENT.register(__ENVS[i])
    return __AGENT.run_episode()['return']

def get_util(policy, envs, n_jobs=None):
    if n_jobs is None:
        return evaluate(policy, envs).util.mean()
    else:
        from joblib import Parallel, delayed
        global __ENVS
        global __CHUNKS
        __ENVS = envs
        __AGENT.register(policy)
        jobs = (delayed(eval_one)(i) for i in range(len(envs)))
        return np.mean(Parallel(n_jobs=n_jobs)(jobs))

def evaluate(policy, envs):
    agent = Agent()
    def run_env(policy, env):
        agent.register(policy)
        agent.register(env)
        tr = agent.run_episode()
        return {'util': tr['return'],
                'observations': len(tr['actions']) - 1}
    return pd.DataFrame(run_env(policy, env) for env in envs)


def bo_policy(envs, max_cost=10., normalize_voi=True, n_calls=60, 
              verbose=False, return_result=False, n_jobs=None):

    def x2theta(x):
        assert len(x) == 4
        cost_weight = x[0]
        voi_weights = np.array(x[1:])
        if normalize_voi:
            voi_weights /= (voi_weights.sum() or 1)
        term_weight = 1
        return np.array([cost_weight, *voi_weights, term_weight])
    
    def objective(x):
        theta = x2theta(x)
        
        with Timer() as t:
            util = get_util(LiederPolicy(theta), envs, n_jobs)
        if verbose:
            print(theta.round(3), '->', round(util, 3),
                  'in', round(t.elapsed), 'sec')
        return - util

    bounds = [ (1., max_cost)] + [ (0., 1.) ] * 3
    with Timer() as t:
        result = gp_minimize(objective, bounds, n_calls=n_calls, random_state=0)

    theta = x2theta(result.x)
    print('BO:', theta.round(3), '->', round(-result.fun, 3),
          'in', round(t.elapsed), 'sec')
    if return_result:
        return LiederPolicy(theta), result
    else:
        return LiederPolicy(theta)


from exact import solve
from policies import FunctionPolicy

def optimal_policy(env, return_value=False):
    with Timer() as t:
        Q, V, pi, info = solve(env)
        v = V(env.init)
    print('optimal -> {} in {} sec'.format(v, t.elapsed))

    policy = FunctionPolicy(pi)
    if return_value:
        return policy, v
    else:
        return policy

def blinkered_policy(env):
    with Timer() as t:
        Q, V, pi, info = solve(env, blinkered=True)
        v = V(env.init)
    print('blinkered -> {} in {} sec'.format(v, t.elapsed))

    policy = FunctionPolicy(pi)
    return policy

def evaluate_many(policies, envs):
    def dfs():
        for name, policy in policies.items():
            df = evaluate(policy, envs)
            df['agent'] = name
            yield df
    return pd.concat(dfs())
