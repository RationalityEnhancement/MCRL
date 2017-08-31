from skopt import gp_minimize
import pandas as pd
import numpy as np

from agents import Agent
from policies import MouselabPolicy, FixedPlanPolicy
from utils import timed



def evaluate(policy, envs):
    agent = Agent()
    def run_env(policy, env):
        agent.register(policy)
        agent.register(env)
        tr = agent.run_episode()
        return {'util': tr['return'],
                'observations': len(tr['actions']) - 1}
    return pd.DataFrame(run_env(policy, env) for env in envs)

    
@timed
def bo_policy(envs, n_calls=60, verbose=False):

    def x2theta(x):
        assert len(x) == 4
        cost_weight = x[0]
        voi_weights = np.array(x[1:])
        voi_weights /= voi_weights.sum()
        term_weight = 1
        return np.array([cost_weight, *voi_weights, term_weight])
        
    def objective(x):
        theta = x2theta(x)
        util = evaluate(MouselabPolicy(theta), envs).util.mean()
        if verbose:
            print(theta.round(3), util)
        return - util
    
    bounds = [ (1., 3.)] + [ (0., 1.) ] * 3
    result = gp_minimize(objective, bounds, n_calls=n_calls, random_state=0)
    theta = x2theta(result.x)
    print('theta:', theta.round(3))
    return MouselabPolicy(theta)


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

    Q, V, pi, info = solve(env, hash_state=hash_state)
    policy = FunctionPolicy(pi)
    if return_value:
        return policy, V(env.init)
    else:
        return policy

# Work in Progress
def data():
    def dfs():
        costs = np.logspace(-7, 0, 8, base=2)
        for cost in costs:
            depth = 2
            sigma = 1
            print('cost', cost)
            train_envs = make_envs(depth, cost, sigma, n=200)
            term = train_envs[0].term_action
            policies = {
                'BO': bo_policy(train_envs, n_calls=30),
                'myopic': MouselabPolicy([1, 1, 0, 0, 1]),
                'full': FixedPlanPolicy(range(1, term+1)),
                'none': FixedPlanPolicy([term]),
            }
            eval_envs = make_envs(depth, cost, sigma, n=200)
            for name, policy in policies.items():
                df = evaluate(policy, envs=eval_envs)
                df['agent'] = name
                df['cost'] = cost
                # print('======= {} ======='.format(name))
                # print('Observations: {:.3f}'.format(df.observations.mean() - 2))
                # print('Return: {:.3f}'.format(df.util.mean() / sigma))
                yield df
    return pd.concat(dfs())




