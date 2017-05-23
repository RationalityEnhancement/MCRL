import itertools as it
import numpy as np


# ---------- Functional utils ---------- #
from toolz import pipe
from toolz.curried import *
max = curry(max)
min = curry(min)
call = lambda f: f()
@curry
def attr(name, obj):
    return getattr(obj, name)
@curry
def invoke(name, obj):
    return getattr(obj, name)()

lmap = curry(compose(list, map))
amap = curry(compose(np.array, lmap))

# ---------- Other ---------- #
def str_join(args, sep=' '):
    return sep.join(map(str, args))

def dict_product(d):
    """All possible combinations of values in lists in `d`"""
    for k, v in d.items():
        if not isinstance(v, list):
            d[k] = [v]

    for v in list(it.product(*d.values())):
        yield dict(zip(d.keys(), v))

def value_iteration(env, discount=0.99, epsilon=0.001, max_iters=100000):
    """Returns the optimal value table for env."""
    V1 = np.zeros(env.observation_space.n)
    if discount == 1:
        threshold = epsilon
    else:
        threshold = epsilon * (1 - discount) / discount

    def value(result):
        # [(0.5, 0, 0, False), (0.5, 0, 0, False)] -> float
        return sum(p * (r + discount * V[s1])
                   for p, s1, r, _ in result)

    for i in range(1, max_iters+1):
        V = V1.copy()
        delta = 0
        for state, actions in env.P.items():

            # Example actions object. Keys are actions, values are
            # lists of (prob, next_state, reward, done).
            # {0: [(1.0, 1, 0, False)],
            #  1: [(1.0, 2, 0, False)],
            #  2: [(1.0, 3, -1, False)],
            #  3: [(1.0, 0, 4, False)]}

            results = actions.values()
            V1[state] = max(map(value, results))
            delta = max(delta, abs(V1[state] - V[state]))

        if delta < threshold:
            return V

    print('NOT CONVERGED')
    return V

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


