from skopt import gp_minimize
import pandas as pd
import numpy as np

from agents import Agent
from policies import LiederPolicy, FixedPlanPolicy, MaxQPolicy
from value_functions import LiederQ
from contexttimer import Timer

from toolz import partition_all
from joblib import Parallel, delayed

from utils import cum_returns

__ENVS = None
__AGENT = Agent()
__CHUNKS = None

def eval_one(i):
    __AGENT.register(__ENVS[i])
    return __AGENT.run_episode()['return']

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
        jobs = (delayed(eval_chunk)(i, return_mean) for i in range(len(__CHUNKS)))
        result = parallel(jobs)
        if return_mean:
            return np.mean(result)
        else:
            return np.concatenate(result)


def get_q_error(theta, envs, parallel=None):
    agent = Agent()
    Q = LiederQ(theta)
    agent.register(Q)
    agent.register(MaxQPolicy(Q))
    err = 0
    for e in envs:
        agent.register(e)
        agent.run_episode()
        tr = agent.ep_trace
        q = cum_returns(tr['rewards'])
        q_hat = np.array([Q.predictOne(s, a) for s, a in zip(tr['states'], tr['actions'])])
        err += np.sum((q - q_hat) ** 2)
    return err

def evaluate(policy, envs):
    agent = Agent()
    def run_env(policy, env):
        agent.register(env)
        agent.register(policy)
        tr = agent.run_episode()
        return {'util': tr['return'],
                'observations': len(tr['actions']) - 1}
    return pd.DataFrame(run_env(policy, env) for env in envs)


def x2theta(x, normalize_voi):
    assert len(x) == 4
    cost_weight = x[0]
    voi_weights = np.array(x[1:])
    if normalize_voi:
        voi_weights /= (voi_weights.sum() or 1)
    term_weight = 1
    return np.array([cost_weight, *voi_weights, term_weight])

def bo_policy(envs, max_cost=10., normalize_voi=True, n_calls=60,
              verbose=False, return_result=False, n_jobs=None, q_learning=False,
              **kwargs):

    if n_jobs is not None:
        parallel = Parallel(n_jobs=n_jobs)
    else:
        parallel = None

    if q_learning:
        def objective(x):
            theta = x2theta(x, normalize_voi)

            with Timer() as t:
                error = get_q_error(theta, envs, parallel)
            if verbose:
                print(theta.round(3), '->', round(error, 3),
                      'in', round(t.elapsed), 'sec')
            return error
    else:
        def objective(x):
            theta = x2theta(x, normalize_voi)

            with Timer() as t:
                util = get_util(LiederPolicy(theta), envs, parallel)
            if verbose:
                print(theta.round(3), '->', round(util, 3),
                      'in', round(t.elapsed), 'sec')
            return - util

    bounds = [ (1., max_cost)] + [ (0., 1.) ] * 3
    with Timer() as t:
        result = gp_minimize(objective, bounds, n_calls=n_calls, random_state=0, **kwargs)

    # gp = result.models[-1]
    # pred_util = - gp.predict(result.x_iters)
    # best_x = result.x_iters[pred_util.argmax()]
    # theta = x2theta(best_x, normalize_voi)
    theta = np.array(x2theta(result.x, normalize_voi))
    util = -result.fun

    print('BO:', theta.round(3), '->', round(util, 3),
          'in', round(t.elapsed), 'sec')
    if return_result:
        return LiederPolicy(theta), result
    else:
        return LiederPolicy(theta)


from exact import solve
from policies import FunctionPolicy


# def _key(args, kwargs):
#     env = args[0]
#     return (env.height, env.cost, bool(kwargs))

# from toolz import memoize

# @memoize(key=_key)
def optimal_policy(env, return_value=False, verbose=False):
    with Timer() as t:
        Q, V, pi, info = solve(env)
        v = V(env.init)
    if verbose:
        print('optimal -> {:.2f} in {:.3f} sec'.format(v, t.elapsed))

    policy = FunctionPolicy(pi)
    if return_value:
        return policy, v
    else:
        return policy

def blinkered_policy(env, kind='standard', verbose=False):
    with Timer() as t:
        Q, V, pi, info = solve(env, blinkered=kind)
        v = V(env.init)
    if verbose:
        print('{} blinkered -> {:.2f} in {:.3f} sec'.format(kind, v, t.elapsed))

    policy = FunctionPolicy(pi)
    return policy

def evaluate_many(policies, envs):
    def dfs():
        for name, policy in policies.items():
            df = evaluate(policy, envs)
            df['agent'] = name
            yield df
    return pd.concat(dfs())
