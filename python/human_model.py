from skopt import gp_minimize
import pandas as pd
import numpy as np

from evaluation import get_util
from policies import Policy
from contexttimer import Timer
from joblib import Parallel, delayed
from toolz import curry



class HumanPolicy(Policy):
    def __init__(self, theta):
        super().__init__()
        self.theta = np.array(theta)

    def attach(self, agent):
        super().attach(agent)
        self.max_paths = len(list(self.env.all_paths(start=1)))
        
    def act(self, state):
        return max(self.env.actions(state), key=self.Q(state))
    
    @curry
    def Q(self, state, action):
        if action == self.env.term_action:
            return self.env.expected_term_reward(self.env._state)
        else:
            return np.dot(self.theta, np.r_[1, self.click_features(state, action)])

    @curry
    def path_features(self, state, path):
        val = self.env.node_value_to(path[-1], state=state)
        return [val.mean, val.var, val.mean * val.var]

    def click_features(self, state, click):
        pfs = sorted(map(self.path_features(state), self.env.all_paths(start=click)))
        while len(pfs) < self.max_paths:  # inneficient
            pfs.append([0, 0, 0])
        return np.concatenate(pfs)

    @classmethod
    def optimize(cls, envs, n_jobs=None, verbose=False, **kwargs):
        if n_jobs is not None:
            parallel = Parallel(n_jobs=n_jobs)
        else:
            parallel = None

        def objective(theta):
            with Timer() as t:
                util = get_util(cls(theta), envs, parallel)
            if verbose:
                print(np.array(theta).round(3), '->', round(util, 3),
                      'in', round(t.elapsed), 'sec')
            return - util

        bounds = [(-10., 10.)] * 7
        with Timer() as t:
            result = gp_minimize(objective, bounds, **kwargs)
        theta = np.array(result.x)
        util = -result.fun

        print('BO:', theta.round(3), '->', round(util, 3),
              'in', round(t.elapsed), 'sec')
        return cls(theta)
