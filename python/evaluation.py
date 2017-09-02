from skopt import gp_minimize
import pandas as pd
import numpy as np

from agents import Agent
from policies import LiederPolicy, FixedPlanPolicy
from contexttimer import Timer
from utils import timed

from joblib import Parallel, delayed
__ENVS = None
__AGENT = Agent()

def eval_one(i): 
    __AGENT.register(__ENVS[i])
    return __AGENT.run_episode()['return']


def evaluate(policy, envs, parallel=None):
    agent = Agent()
    def run_env(policy, env):
        agent.register(policy)
        agent.register(env)
        tr = agent.run_episode()
        return {'util': tr['return'],
                'observations': len(tr['actions']) - 1}
    return pd.DataFrame(run_env(policy, env) for env in envs)

def bo_policy(envs, max_cost=10., normalize_voi=True, n_calls=60, 
              verbose=False, return_result=False, n_jobs=1):

    def x2theta(x):
        assert len(x) == 4
        cost_weight = x[0]
        voi_weights = np.array(x[1:])
        if normalize_voi:
            voi_weights /= (voi_weights.sum() or 1)
        term_weight = 1
        return np.array([cost_weight, *voi_weights, term_weight])
    
    if n_jobs > 1:
        global __ENVS, __AGENT
        __ENVS = envs
        parallel = Parallel(n_jobs=n_jobs)

        def objective(x):
            theta = x2theta(x)
            __AGENT.register(LiederPolicy(theta))
            with Timer() as t:
                util = np.mean(parallel(delayed(eval_one)(i) for i in range(len(envs))))
            if verbose:
                print(theta.round(3), '->', round(util, 3),
                      'in', round(t.elapsed), 'sec')
            return - util
    else:
        def objective(x):
            theta = x2theta(x)
            with Timer() as t:
                util = evaluate(LiederPolicy(theta), envs).util.mean()
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

@timed
def optimal_policy(env, return_value=False):
    if hasattr(env, 'n_arm'):
        hash_state = 'sort'
    elif hasattr(env, 'tree'):
        hash_state = 'sort_tree'
    else:
        raise ValueError('Not a recognized environment.')

    with Timer() as t:
        Q, V, pi, info = solve(env, hash_state=hash_state)
        v = V(env.init)
    print('optimal -> {} in {} sec'.format(v, t.elapsed))

    policy = FunctionPolicy(pi)
    if return_value:
        return policy, v
    else:
        return policy

def evaluate_many(policies, envs):
    def dfs():
        for name, policy in policies.items():
            df = evaluate(policy, envs)
            df['agent'] = name
            yield df
    return pd.concat(dfs())


if __name__ == '__main__':
    from mouselab import MouselabEnv
    envs = [MouselabEnv() for _ in range(4)]
    pol = bo_policy(envs, n_calls=20, verbose=True, n_jobs=4)

