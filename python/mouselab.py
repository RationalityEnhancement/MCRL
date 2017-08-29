from collections import namedtuple, defaultdict, deque, Counter
import numpy as np
import gym
from toolz import memoize, curry
from functools import lru_cache
from gym import spaces
import itertools as it
from distributions import *

class MouselabEnv(gym.Env):
    """MetaMDP for a tree with a discrete unobserved reward function."""
    metadata = {'render.modes': ['human', 'array']}
    term_state = '__term_state__'
    def __init__(self, branch=2, height=2, reward=None, cost=0, ground_truth=None):
        self.branch = branch
        if hasattr(self.branch, '__len__'):
            self.height = len(self.branch)
        else:
            self.height = height
            self.branch = [self.branch] * self.height

        self.cost = - abs(cost)
        self.reward = reward if reward is not None else Normal(1, 1)
        self.max = cmax if hasattr(reward, 'vals') else dmax
        self.ground_truth = np.array(ground_truth) if ground_truth is not None else None


        self.tree = self._build_tree()
        self.init = (0,) + (self.reward,) * (len(self.tree) - 1)
        self.term_action = len(self.tree)
        self.reset()

        self.action_space = spaces.Discrete(len(self.tree) + 1)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=len(self.tree))
        self.subtrees = self._get_subtrees()

    def _reset(self):
        self._state = self.init
        return self.features(self._state)

    def _step(self, action):
        if self._state is self.term_state:
            assert 0, 'state is terminal'
            # return None, 0, True, {}
        if action == self.term_action:
            # self._state = self.term_state
            if self.ground_truth is not None:
                reward = self.term_reward().expectation()
                # path = list(self.best_path())
                # reward = self.ground_truth[path].sum()
            else:
                reward = self.term_reward().expectation()
                # reward = self.term_reward().sample()
            done = True
        elif self._state[action] is not self.reward:  # already observed
            reward = 0
            done = False
        else:  # observe a new node
            self._state = self._observe(action)
            reward = self.cost
            done = False
        return self.features(self._state), reward, done, {}

    def _observe(self, action):
        if self.ground_truth is not None:
            result = self.ground_truth[action]
        else:
            result = self._state[action].sample()
        s = list(self._state)
        s[action] = result
        return tuple(s)

    def actions(self, state):
        """Yields actions that can be taken in the given state.

        Actions include observing the value of each unobserved node and terminating.
        """
        if state is self.term_state:
            return
        for i, v in enumerate(state):
            if hasattr(v, 'sample'):
                yield i
        yield self.term_action

    def results(self, state, action):
        """Returns a list of possible results of taking action in state.

        Each outcome is (probability, next_state, reward).
        """
        if action == self.term_action:
            # R = self.term_reward()
            # S1 = Categorical([self.term_state])
            # return cross(S1, R)
            yield (1, self.term_state, self.expected_term_reward(state))
        else:
            for r, p in state[action]:
                s1 = list(state)
                s1[action] = r
                yield (p, tuple(s1), self.cost)

    def features(self, state=None):
        state = state if state is not None else self._state
        return state

        # if state is None:
        #     return np.full(len(self.tree), np.nan)
        # # Is each node observed?
        # return np.array([1. if hasattr(x, 'sample') else 0.
        #                  for x in state])
    
    def action_features(self, action, state=None):
        state = state if state is not None else self._state
        assert state is not None

        # if action == self.term_action:
        #     tr_mu, tr_sigma = norm.fit(self.term_reward.sample(10000))
        #     return np.r_[0, 0, 0, 0, 0, tr_mu, tr_sigma]
        # nq_mu, nq_sigma = norm.fit(self.node_quality(action).sample(10000))
        # nqpi_mu, nqpi_sigma = norm.fit(self.node_quality(action).sample(10000))
        # return np.r_[1, nq_mu, nq_sigma, nqpi_mu, nqpi_sigma, 0, 0]

        if action == self.term_action:
            return np.array([
                0,
                0,
                0,
                0
            ])

        return np.array([
            self.cost,
            self.myopic_voc(action, state),
            self.vpi_action(action, state),
            self.vpi(state),
            self.expected_term_reward(state)
        ])

    def subtree(self, node):
        # return state[self.subtrees[node]]
        def gen(n):
            yield n
            for n1 in self.tree[n]:
                yield from gen(n1)
        return tuple(gen(node))

    def term_reward(self, state=None):
        state = state if state is not None else self._state
        assert state is not None
        return self.node_value(0, state)

    def best_path(self, state=None):
        state = state if state is not None else self._state
        n = 0
        while self.tree[n]:
            n = max(self.tree[n],
                    key=lambda n1: self.node_quality(n1, state).expectation())
            yield n

    @lru_cache(maxsize=100000)
    def expected_term_reward(self, state):
        return self.term_reward(state).expectation()

    def node_value(self, node, state=None):
        """A distribution over total rewards after the given node."""
        state = state if state is not None else self._state
        return max((self.node_value(n1, state) + state[n1]
                    for n1 in self.tree[node]), 
                   default=PointMass(0), key=expectation)

    def node_value_to(self, node, state=None):
        """A distribution over rewards up to and including the given node."""
        state = state if state is not None else self._state
        start_value = PointMass(0)
        return sum((state[n] for n in self.path_to(node)), start_value)

    def node_quality(self, node, state=None):
        """A distribution of total expected rewards if this node is visited."""
        state = state if state is not None else self._state
        return self.node_value_to(node, state) + self.node_value(node, state)

    @lru_cache(maxsize=100000)
    def myopic_voc(self, action, state):
        return (self.node_value_after_observe((action,), 0, state).expectation()
                - self.expected_term_reward(state)
                )

    @lru_cache(maxsize=100000)
    def vpi_action(self, action, state):
        obs = self._relevant_subtree(action)
        return (self.node_value_after_observe(obs, 0, state).expectation()
                - self.expected_term_reward(state)
                )

    @lru_cache(maxsize=100000)
    def vpi(self, state):
        obs = self.subtree(0)
        return (self.node_value_after_observe(obs, 0, state).expectation()
                - self.expected_term_reward(state)
                )

    def unclicked(self, state):
        return sum(1 for x in state if hasattr(x, 'sample'))

    @memoize
    def _relevant_subtree(self, node):
        trees = [self.subtree(n1) for n1 in self.tree[0]]
        for t in trees:
            if node in t:
                return tuple(t)
        assert False


    # # @memoize
    def node_value_after_observe(self, obs, node, state):
        """A distribution over the expected value of node, after making an observation.
        
        obs can be a single node, a list of nodes, or 'all'
        """
        obs_tree = self.to_tree(state, node, obs)
        return node_value_after_observe(self.max, self.reward, obs_tree)

    # @memoize
    # def node_value_after_observe(self, obs, node, state):
    #     """A distribution over the expected value of node, after making an observation.
        
    #     obs can be a single node, a list of nodes, or 'all'
    #     """
    #     def r(n):
    #         if obs == 'all' or n in obs:
    #             return state[n]
    #         else:
    #             return expectation(state[n])

    #     return dmax((self.node_value_after_observe(obs, n1, state) + r(n1)
    #                  for n1 in self.tree[node]),
    #                 default=PointMass(0))

    def path_to(self, node, start=0):
        path = [start]
        if node == start:
            return path
        for _ in range(self.height + 1):
            children = self.tree[path[-1]]
            for i, child in enumerate(children):
                if child == node:
                    path.append(node)
                    return path
                if child > node:
                    path.append(children[i-1])
                    break
            else:
                path.append(child)
        assert False

    def all_paths(self, start=0):
        def rec(path):
            children = self.tree[path[-1]]
            if children:
                for child in children:
                    yield from rec(path + [child])
            else:
                yield path

        return rec([start])

    def _get_subtrees(self):
        slices = [0] * len(self.tree)
        def get_end(n):
            end = max((get_end(n1) for n1 in self.tree[n]), default=n+1)
            slices[n] = slice(n, end)
            return end
        get_end(0)
        return slices


    def _build_tree(self):
        # num_node = np.cumsum(self.branch).sum() + 1
        def nodes_per_layer():
            n = 1
            yield n
            for b in self.branch:
                n *= b
                yield n

        num_node = sum(nodes_per_layer())
        T = [[] for _ in range(num_node)]  # T[i] = [c1, c2, ...] or [] if i is terminal

        ids = it.count(0)
        def expand(i, d):
            if d == self.height:
                return
            for _ in range(self.branch[d]):
                next_i = next(ids)
                T[i].append(next_i)
                expand(next_i, d+1)

        expand(next(ids), 0)
        return T

    def _render(self, mode='notebook', close=False):
        from graphviz import Digraph
        from IPython.display import display
        import matplotlib as mpl
        from matplotlib.colors import rgb2hex
        if close:
            return
        
        vmin = -2
        vmax = 2
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = mpl.cm.get_cmap('RdYlGn')
        colormap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        colormap.set_array(np.array([vmin, vmax]))

        def color(val):
            if val > 0:
                return '#8EBF87'
            else:
                return '#F7BDC4'
        
        dot = Digraph()
        for x, ys in enumerate(self.tree):
            r = self._state[x]
            observed = not hasattr(self._state[x], 'sample')
            c = color(r) if observed else 'grey'
            l = str(round(r, 2)) if observed else str(x)
            dot.node(str(x), label=l, style='filled', color=c)
            for y in ys:
                dot.edge(str(x), str(y))
        display(dot)


    def to_tree(self, state, n, obs=()):
        val = 'obs' if (hasattr(state[n], 'sample') and n in obs) else state[n]
        children = tuple(self.to_tree(state, c, obs) for c in self.tree[n])
        return (val, children)


@memoize
def node_value_after_observe(_max, reward, obs_tree):
    """A distribution over the expected value of node, after making an observation.
    
    obs can be a single node, a list of nodes, or 'all'
    """
    def subjective_reward(x):
        if x == 'obs':
            return reward
        else:
            return expectation(x)

    def rec(tree):
        return _max((rec(c) + subjective_reward(c[0])
                     for c in tree[1]),
                    default=PointMass(0))
    return rec(obs_tree)

