import numpy as np
import itertools as it
from collections import Counter, defaultdict, deque
from toolz.curried import *


class MetaTree(object):
    """MetaMDP for a tree with a discrete reward function."""
    unknown = np.inf
    term_state = None
    def __init__(self, branch=2, height=2, outcomes=(0,1), cost=0):
        self.branch = branch
        self.height = height
        self.outcomes = outcomes
        self.cost = cost
        self.tree = self._build_tree()
        self.init = (self.unknown, ) * len(self.tree)
        self.term_action = len(self.tree)
        self.outcome_prob = 1 / len(outcomes)
        self.expected_outcome = sum(outcomes) / len(outcomes)

    def actions(self, state):
        """Yields actions that can be taken in the given state.

        Actions include observing the value of each unobserved node and terminating.
        """
        for i, v in enumerate(state):
            if v is self.unknown and i > 0:
                yield i
        yield self.term_action

    def results(self, state, action):
        """Returns a list of possible results of taking action in state.

        Each outcome is (probability, next_state, reward).
        """
        if action == self.term_action:
            r = self.term_value(state)
            return [(1, self.term_state, r)]
        else:
            assert state[action] is self.unknown
            result = []
            for o in self.outcomes:
                s1 = list(state)
                s1[action] = o
                result.append((self.outcome_prob, tuple(s1), self.cost))
            return result

    @memoize
    def term_value(self, state):
        """Returns the expected value of terminating computation given a belief state.

        This is equal to the value of the initial state in the object-level MDP.
        """
        return self.tree_V(state, 0)

    def tree_V(self, state, s):
        # includes the reward attained at state s (it's not really a value function)
        r = state[s]
        if r is self.unknown:
            r = self.expected_outcome
        future_reward = max((self.tree_V(state, s1) for s1 in self.tree[s]), default=0)
        return r + future_reward

    def subtree(self, state, n):
        """Returns the substree of the belief state with root n."""
        if not self.tree[n]:  # leaf node
            return state[n]
        c1, c2 = self.tree[n]
        return tuple(state[i] for i in range(n, 2 * c2 - c1))

    def _build_tree(self):
        """Constructs the transition object-level MDP."""
        num_node = self.branch ** (self.height+1) - 1
        T = [[] for _ in range(num_node)]  # T[i] = [c1, c2] or [] if i is terminal
        ids = it.count(0)
        def expand(i, d):
            if d == self.height:
                return
            for _ in range(self.branch):
                next_i = next(ids)
                T[i].append(next_i)
                expand(next_i, d+1)

        expand(next(ids), 0)
        return T

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



