from collections import namedtuple, defaultdict, deque, Counter
import numpy as np
import gym
from gym import spaces
from distributions import smax, cmax, sample, expectation, Normal, PointMass
from toolz import memoize, get
import random
from contracts import contract


NO_CACHE = False
if NO_CACHE:
    lru_cache = lambda _: (lambda f: f)
else:
    from functools import lru_cache

CACHE_SIZE = int(2**20)
SMALL_CACHE_SIZE = int(2**14)
ZERO = PointMass(0)

class MouselabEnv(gym.Env):
    """MetaMDP for a tree with a discrete unobserved reward function."""
    metadata = {'render.modes': ['human', 'array']}
    term_state = '__term_state__'

    def __init__(self, tree, init, ground_truth=None, cost=0, term_belief=True, sample_term_reward=False):
        self.tree = tree
        self.init = (0, *init[1:])
        
        if ground_truth is False:
            self.ground_truth = ()
        elif ground_truth is not None:
            if len(ground_truth) != len(init):
                print(ground_truth)
                print(init)
                raise ValueError('len(ground_truth) != len(init)')
            self.ground_truth = np.array(ground_truth)
            self.ground_truth[0] = 0.
        else:
            self.ground_truth = np.array(list(map(sample, init)))
            self.ground_truth[0] = 0.
        self.cost = - abs(cost)
        self.term_belief = term_belief
        self.sample_term_reward = sample_term_reward
        self.term_action = len(self.init)

        # Required for gym.Env API.
        self.action_space = spaces.Discrete(len(self.init) + 1)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=len(self.init))

        self.initial_states = None  # TODO
        self.exact = True  # TODO
        
        self.subtree = self._get_subtree()
        self.subtree_slices = self._get_subtree_slices()
        self.paths = self.get_paths(0)
        self.reset()

        self._hash = hash((str(self.tree), self.init, str(list(self.ground_truth))))

    def __hash__(self):
        return self._hash

    def _reset(self):
        if self.initial_states:
            self.init = random.choice(self.initial_states)
        self._state = self.init
        return self._state

    def _step(self, action):
        if self._state is self.term_state:
            assert 0, 'state is terminal'
        if action == self.term_action:
            reward = self._term_reward()
            self._state = self.term_state
            done = True
        elif not hasattr(self._state[action], 'sample'):  # already observed
            assert 0, self._state[action]
            reward = 0
            done = False
        else:  # observe a new node
            self._state = self._observe(action)
            reward = self.cost
            done = False
        return self._state, reward, done, {}

    def _term_reward(self):
        if self.term_belief:
            return self.expected_term_reward(self._state)

        returns = [self.ground_truth[list(path)].sum() 
                   for path in self.optimal_paths()]
        if self.sample_term_reward:
            return np.random.sample(returns)
        else:
            return np.mean(returns)

    def _observe(self, action):
        if self.ground_truth is ():
            result = self._state[action].sample()
        else:
            result = self.ground_truth[action]
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
            yield (1, self.term_state, self.expected_term_reward(state))
        else:
            for r, p in state[action]:
                s1 = list(state)
                s1[action] = r
                yield (p, tuple(s1), self.cost)
    
    def action_features(self, action, state=None):
        state = state if state is not None else self._state
        assert state is not None


        if action == self.term_action:
            return np.array([
                0,
                0,
                0,
                0,
                self.expected_term_reward(state)
            ])

        return np.array([
            self.cost,
            self.myopic_voc(action, state),
            self.vpi_action(action, state),
            self.vpi(state),
            self.expected_term_reward(state)
        ])


    def term_reward(self, state=None):
        """A distribution over the return gained by acting given a belief state."""
        state = state if state is not None else self._state
        return self.node_value(0, state)


    def optimal_paths(self, state=None, tolerance=0.01):
        state = state if state is not None else self._state
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
        return max((self.node_value(n1, state) + state[n1]
                    for n1 in self.tree[node]), 
                   default=ZERO, key=expectation)
    
    def node_value_to(self, node, state=None):
        """A distribution over rewards up to and including the given node."""
        state = state if state is not None else self._state
        start_value = ZERO
        return sum((state[n] for n in self.path_to(node)), start_value)

    def node_quality(self, node, state=None):
        """A distribution of total expected rewards if this node is visited."""
        state = state if state is not None else self._state
        return self.node_value_to(node, state) + self.node_value(node, state)

    # @lru_cache(CACHE_SIZE)
    @contract
    def myopic_voc(self, action, state) -> 'float, >= -0.001':
        return (self.node_value_after_observe((action,), 0, state).expectation()
                - self.expected_term_reward(state)
                )

    # @lru_cache(CACHE_SIZE)
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

    @contract
    def vpi(self, state) -> 'float, >= -0.001':
        obs = self.subtree[0]
        return (self.node_value_after_observe(obs, 0, state).expectation()
                - self.expected_term_reward(state)
                )

    def unclicked(self, state):
        return sum(1 for x in state if hasattr(x, 'sample'))

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
    
    def mean_Q(self, node):
        r = self.ground_truth[node]
        lst = [self.mean_Q(n1) for n1 in self.tree[node]]
        if lst:
            return r+np.mean(lst)
        return r 
    
    def get_paths(self, node):  
        if self.tree[node] == []:
            return [[]]
        paths = []
        for n in self.tree[node]:
            new_paths = self.get_paths(n)
            for path in new_paths:
                path.insert(0, n)
                paths.append(path)
        return paths
    
    @lru_cache(None) 
    def _relevant_subtree(self, node):
        trees = [self.subtree[n1] for n1 in self.tree[0]]
        for t in trees:
            if node in t:
                return tuple(t)
        assert False

    @lru_cache(None) 
    def leaves(self):
        return [path[-1] for path in self.all_paths()]

    def path_values(self, state):
        return [self.node_quality(node, state) 
                for node in self.leaves()]

    def node_value_after_observe(self, obs, node, state):
        """A distribution over the expected value of node, after making an observation.
        
        obs can be a single node, a list of nodes, or 'all'
        """
        obs_tree = self.to_obs_tree(state, node, obs)
        if self.exact:
            return exact_node_value_after_observe(obs_tree)
        else:
            return node_value_after_observe(obs_tree)

    @memoize
    def path_to(self, node, start=0):
        path = [start]
        if node == start:
            return path
        for _ in range(10000):
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

    @memoize
    def all_paths(self, start=0):
        def rec(path):
            children = self.tree[path[-1]]
            if children:
                for child in children:
                    yield from rec(path + [child])
            else:
                yield path

        return list(rec([start]))

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

    @classmethod
    def new_symmetric(cls, branching, reward, seed=None, **kwargs):
        """Returns a MouselabEnv with a symmetric structure."""
        if seed is not None:
            np.random.seed(seed)
        if not callable(reward):
            r = reward
            reward = lambda depth: r

        init = []
        tree = []

        def expand(d):
            my_idx = len(init)
            init.append(reward(d))
            children = []
            tree.append(children)
            for _ in range(get(d, branching, 0)):
                child_idx = expand(d+1)
                children.append(child_idx)
            return my_idx

        expand(0)
        return cls(tree, init, **kwargs)

    def _render(self, mode='notebook', close=False):
        if close:
            return
        from graphviz import Digraph
        
        
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
        return dot


    def to_obs_tree(self, state, node, obs=(), sort=True):
        maybe_sort = sorted if sort else lambda x: x
        def rec(n):
            subjective_reward = state[n] if n in obs else expectation(state[n])
            children = tuple(maybe_sort(rec(c) for c in self.tree[n]))
            return (subjective_reward, children)
        return rec(node)

@lru_cache(SMALL_CACHE_SIZE)
def node_value_after_observe(obs_tree):
    """A distribution over the expected value of node, after making an observation.
    
    `obs` can be a single node, a list of nodes, or 'all'
    """
    children = tuple(node_value_after_observe(c) + c[0] for c in obs_tree[1])
    return smax(children, default=ZERO)


@lru_cache(CACHE_SIZE)
def exact_node_value_after_observe(obs_tree):
    """A distribution over the expected value of node, after making an observation.
    
    `obs` can be a single node, a list of nodes, or 'all'
    """
    children = tuple(exact_node_value_after_observe(c) + c[0]
                     for c in obs_tree[1])
    return cmax(children, default=ZERO)

   
