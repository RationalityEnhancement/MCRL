from collections import namedtuple, defaultdict, deque, Counter
import numpy as np
import gym
from gym import spaces
import itertools as it
from distributions import cmax, smax, expectation, Normal, PointMass, ZERO
from toolz import memoize, get, sliding_window, concatv, curry
import random
from contracts import contract
from tree import TreeIndex


NO_CACHE = False
if NO_CACHE:
    lru_cache = lambda _: (lambda f: f)
else:
    from functools import lru_cache

CACHE_SIZE = int(2**16)
SMALL_CACHE_SIZE = int(2**14)


max = curry(max)
cmax = curry(cmax)

class MetaTreeEnv(gym.Env):
    """MetaMDP for a tree with a discrete unobserved reward function."""
    metadata = {'render.modes': ['human', 'array']}
    term_state = '__term_state__'
    def __init__(self, init_belief, ground_truth=None, cost=0, sample_term_reward=False):

        self.init_belief = init_belief
        self.ground_truth = ground_truth
        self.cost = - abs(cost)
        self.sample_term_reward = sample_term_reward
        self.term_action = len(self.init_belief)
        self.reset()
        self.index = TreeIndex(init_belief)


        # Required for gym.Env API.
        self.action_space = spaces.Discrete(len(self.init_belief) + 1)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=len(self.init_belief))

    def _reset(self):
        self._state = self.init_belief
        return self._state

    def _step(self, action):
        if self._state is self.term_state:
            assert 0, 'state is terminal'

        if action == self.term_action:
            self._state = self.term_state
            reward = self._get_term_reward()
            done = True

        elif not hasattr(self._state[action], 'sample'):  # already observed
            assert 0, f'{action} has already been observed'
            reward = 0
            done = False

        else:  # observe a new node
            self._state = self._state.update(action, self._observe(action))
            reward = self.cost
            done = False

        return self._state, reward, done, {}

    def _ground_truth_term_reward(self):
        assert self.ground_truth is not None
        paths = self.optimal_paths(self._state)

        # if self.sample_term_reward:
        #     if self.ground_truth is not None:
        #         reward = self.ground_truth[list(path)].sum()
        #     else:
        #         reward = self.term_reward().sample()
        # else:
        #     if self.ground_truth is not None:
        #         reward =  np.mean([self.ground_truth[list(path)].sum()
        #                            for path in self.optimal_paths()])
            paths = self.optimal_paths(self._state)



        if self.sample_term_reward:
            return self.term_reward(self._state).sample()
        else:
            return self.term_reward(self._state).expectation()
        
        

    def _observe(self, action):
        if self.ground_truth is None:
            return self._state[action].sample()
        else:
            return self.ground_truth[action]

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

    def term_reward(self, state):
        """A distribution over the return gained by acting given a belief state."""
        return self.node_value(0, state)
    
    def optimal_paths(self, state, tolerance=0.01):
        def rec(path):
            children = self.tree[path[-1]]
            if not children:
                yield path
                return
            quals = [self.node_quality(n1, state).expectation()
                     for n1 in children]
            best_q = max(quals)
            for n1, q in zip(children, quals):
                if np.abs(q - best_q) < tolerance:
                    yield from rec(path + (n1,))

        yield from rec((0,))

    @lru_cache(CACHE_SIZE)
    def expected_term_reward(self, state):
        return self.term_reward(state).expectation()

    def node_value(self, node, state=None):
        """A distribution over total rewards after the given node."""
        state = state if state is not None else self._state
        address = self.index[node]
        return state[address].value(max(key=expectation, default=ZERO))
    
    def node_value_to(self, node, state=None):
        """A distribution over rewards up to and including the given node."""
        state = state if state is not None else self._state
        address = self.index[node]
        return sum((n.val for n in state.path(address)), ZERO)

    def node_quality(self, node, state=None):
        """A distribution of total expected rewards if this node is visited."""
        return self.node_value_to(node, state) + self.node_value(node, state)

    @contract
    def myopic_voc(self, action, state) -> 'float, >= -0.001':
        return (self.node_value_after_observe((action,), 0, state).expectation()
                - self.expected_term_reward(state)
                )

    @contract
    def vpi_branch(self, action, state) -> 'float, >= -0.001':
        obs = self._relevant_subtree(action)
        return (self.node_value_after_observe(obs, 0, state).expectation()
                - self.expected_term_reward(state)
                )
    
    @contract
    def vpi_action(self, action, state) -> 'float, >= -0.001':
        obs = (*self.subtree[action][1:], *self.path_to(action)[1:])
        return (self.node_value_after_observe(obs, 0, state).expectation()
                - self.expected_term_reward(state)
                )

    @lru_cache(CACHE_SIZE)
    @contract
    def vpi(self, state) -> 'float, >= -0.001':
        obs = self.subtree[0]
        return (self.node_value_after_observe(obs, 0, state).expectation()
                - self.expected_term_reward(state)
                )

    def true_Q(self, node):
        """The object-level Q function."""
        r = self.ground_truth[node]
        return r + max((self.true_Q(n1) for n1 in self.tree[node]),
                    default=0)
    
    def worst_Q(self, node):
        """The object-level Q function."""
        r = self.ground_truth[node]
        return r + min((self.worst_Q(n1) for n1 in self.tree[node]),
                    default=0)
    
    def rand_Q(self, node):
        """The object-level Q function."""
        r = self.ground_truth[node]
        lst = [self.rand_Q(n1) for n1 in self.tree[node]]
        if lst:
            return r+random.choice(lst)
        return r 
    
    def mean_Q(self,node):
        r = self.ground_truth[node]
        lst = [self.mean_Q(n1) for n1 in self.tree[node]]
        if lst:
            return r+np.mean(lst)
        return r 
    
    @lru_cache(None) 
    def _relevant_subtree(self, node):
        trees = [self.subtree[n1] for n1 in self.tree[0]]
        for t in trees:
            if node in t:
                return tuple(t)
        assert False

    @lru_cache(None) 
    def leaves(self, node):
        trees = [self.subtree[n1] for n1 in self.tree[0]]
        for t in trees:
            if node in t:
                return tuple(t)
        assert False

    def node_value_after_observe(self, obs, node, state):
        """A distribution over the expected value of node, after making an observation.
        
        obs can be a single node, a list of nodes, or 'all'
        """
        if self._binary:
            obs_flat = self.to_obs_flat(state, node, obs)
            if self.exact:
                return exact_flat_node_value_after_observe(obs_flat)
            else:
                return flat_node_value_after_observe(obs_flat)
        else:
            obs_tree = self.to_obs_tree(state, node, obs)
            if self.exact:
                return exact_node_value_after_observe(obs_tree)
            else:
                return node_value_after_observe(obs_tree)

    def _get_subtree_slices(self):
        slices = [0] * len(self.tree)
        def get_end(n):
            end = max((get_end(n1) for n1 in self.tree[n]), default=n+1)
            slices[n] = slice(n, end)
            return end
        get_end(0)
        return slices

    def _get_subtree(self):
        def gen(n):
            yield n
            for n1 in self.tree[n]:
                yield from gen(n1)
        return [tuple(gen(n)) for n in range(len(self.tree))]


    def _render(self, mode='notebook', close=False):
        if close:
            return
        from graphviz import Digraph
        from IPython.display import display
        

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

    def to_obs_tree(self, state, node, obs=(), sort=True):
        maybe_sort = sorted if sort else lambda x: x
        def rec(n):
            subjective_reward = state[n] if n in obs else expectation(state[n])
            children = tuple(maybe_sort(rec(c) for c in self.tree[n]))
            return (subjective_reward, children)
        # return obs_rec(self.tree, state, obs, node)
        return rec(node)


