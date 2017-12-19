from skopt import gp_minimize
import numpy as np

from evaluation import get_util
from policies import Policy
from contexttimer import Timer
from joblib import Parallel, delayed
from toolz import curry, concat

import warnings
warnings.filterwarnings("ignore",
    message="The objective has been evaluated at this point before.")


def softmax(x, temp=1):
    ex = np.exp((x - x.max()) / temp)
    return ex / ex.sum()

def zero_pad(size, x):
    return np.r_[x, np.zeros(size - len(x))]

@curry
def path_features(env, state, path):
    val = env.node_value_to(path[-1], state=state)
    # m = val.mean / (max(env.reward.vals) * len(path))
    # s = val.std / env.node_value_to(path[-1], state=env.init).std
    m = val.mean
    s = val.std
    return [m, s, m * s]

@curry
def action_features(env, state, action, max_paths=2, aggregate=False, w=None):
    term = (action == env.term_action)
    if term:
        x = np.zeros(2 + 3 * (1 if aggregate else max_paths))
        x[0] = env.expected_term_reward(state)
        return x
    if not (term or hasattr(state[action], 'sample')):
        # Clicking an already revealed node, an invalid action
        return np.zeros(2 + max_paths * 3)

    w = w or np.array([1, 0])
    path_fs = list(map(path_features(env, state), env.all_paths(start=action)))
    while len(path_fs) < max_paths:
        path_fs.append([0,0,0])

    node_features = ([] if term else
                     w @ path_fs if aggregate else
                     list(concat(sorted(path_fs))))
    
                             
    return np.r_[
        0,  # term
        1,  # not term
        node_features
     ]
      

class HumanPolicy(Policy):
    def __init__(self, theta, aggregate=False, temp=0):
        super().__init__()
        self.theta = np.array(theta)
        self.temp = temp
        self.aggregate = aggregate

    def attach(self, agent):
        super().attach(agent)
        self.max_paths = len(list(self.env.all_paths(start=1)))
        self._actions = np.arange(self.n_action)
        
    def act(self, state):
        if self.temp == 0:
            return max(self.env.actions(state), key=self.Q(state))
        else:
            probs = self.action_distribution(state)
            return np.random.choice(self._actions, p=probs)

    def action_distribution(self, state):
        q = np.zeros(self.n_action) - 1e5
        for a in self.env.actions(state):
            q[a] = self.Q(state, a)
        # return q
        return softmax(q, self.temp)
    
    @curry
    def Q(self, state, action):
        if self.aggregate:
            if action == self.env.term_action:
                return self.env.expected_term_reward(state)
            path_fs = list(map(path_features(self.env, state), 
                               self.env.all_paths(start=action)))
            return np.dot(path_fs, self.theta).max()

        else:
            return np.dot(self.theta, self.phi(state, action))

    def phi(self, state, action):
        if action == self.env.term_action:
            x = np.zeros(len(self.theta))
            x[0] = self.env.expected_term_reward(state)
            return x


        path_fs = sorted(map(path_features(self.env, state), 
                             self.env.all_paths(start=action)))
        
        while len(path_fs) < self.max_paths:
            path_fs.append([0,0,0])

        return np.array([0, 1, *concat(path_fs)])


    @classmethod
    def optimize(cls, envs, aggregate=False, n_jobs=None, verbose=False, n_calls=60):
        if n_jobs is not None:
            parallel = Parallel(n_jobs=n_jobs)
        else:
            parallel = None

        def objective(x):
            theta = np.r_[1, x]  # term_reward weight fixed to 1
            with Timer() as t:
                util = get_util(cls(theta, aggregate=aggregate), envs, parallel)
            if verbose:
                print(np.array(theta).round(3), '->', round(util, 3),
                      'in', round(t.elapsed), 'sec')
            return - util

        n_path_fs = 3 if aggregate else 6

        bounds = [
            (-5., 5.),                      # is click
            *([(-5., 5.)] * n_path_fs)      # node_features
        ]
        with Timer() as t:
            result = gp_minimize(objective, bounds, n_calls=n_calls)
        theta = np.r_[1, result.x]
        util = -result.fun

        print('BO:', theta.round(3), '->', round(util, 3),
              'in', round(t.elapsed), 'sec')
        return cls(theta, aggregate=aggregate), result
