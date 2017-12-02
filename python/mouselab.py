from collections import namedtuple, defaultdict, deque, Counter
import numpy as np
import gym
from gym import spaces
import itertools as it
from distributions import cmax, smax, expectation, Normal, PointMass
from toolz import memoize
import random
from contracts import contract


NO_CACHE = False
if NO_CACHE:
    lru_cache = lambda _: (lambda f: f)
else:
    from functools import lru_cache

CACHE_SIZE = int(2**16)
SMALL_CACHE_SIZE = int(2**14)
ZERO = PointMass(0)

class MouselabEnv(gym.Env):
    """MetaMDP for a tree with a discrete unobserved reward function."""
    metadata = {'render.modes': ['human', 'array']}
    term_state = '__term_state__'
    def __init__(self, branch=2, height=2, reward=None, cost=0, sample_term_reward=False,
                 ground_truth=None, expand_only=True, initial_states=None):
        self.branch = branch
        self._binary = branch == 2

        if hasattr(self.branch, '__len__'):
            self.height = len(self.branch)
        else:
            self.height = height
            self.branch = [self.branch] * self.height

        if hasattr(reward, 'sample'):
            self.reward = reward if reward is not None else Normal(1, 1)
            self.iid_rewards = True
        else:
            self.iid_rewards = False
        self.cost = - abs(cost)
        self.ground_truth = np.array(ground_truth) if ground_truth is not None else None
        self.tree = self._build_tree()
        
        self.initial_states = initial_states
        self.exact = hasattr(reward, 'vals')
        if self.exact:
            assert self.iid_rewards
            self.max = cmax
            self.init = (0., *((self.reward,) * (len(self.tree) - 1)))
        else:
            # Distributions represented as samples.
            self.max = smax
            # self.init = (0, *(self.reward.to_sampledist() for _ in range(len(self.tree) - 1)))
            self.init = (0., *((self.reward.to_sampledist(),) * (len(self.tree) - 1)))
            # if self.iid_rewards:
                # self.init = (0, *(self.reward.copy() for _ in range(len(self.tree) - 1)))
            # else:u
                # self.init = reward
        # self.expected_term_reward = self.reward.expectation()
        self.sample_term_reward = sample_term_reward
        self.action_space = spaces.Discrete(len(self.tree) + 1)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=len(self.tree))
        self.subtree = self._get_subtree()
        self.subtree_slices = self._get_subtree_slices()
        self.term_action = len(self.tree)
        self.reset()

    def _reset(self):
        if self.initial_states:
            self.init = random.choice(self.initial_states)
        self._state = self.init
        return self.features(self._state)

    def _step(self, action):
        if self._state is self.term_state:
            assert 0, 'state is terminal'
            # return None, 0, True, {}
        if action == self.term_action:
            # self._state = self.term_state
            if self.sample_term_reward:
                if self.ground_truth is not None:
                    path = random.choice(list(self.optimal_paths()))
                    reward = self.ground_truth[list(path)].sum()
                else:
                    reward = self.term_reward().sample()
            else:
                reward = self.term_reward().expectation()
            done = True
        elif not hasattr(self._state[action], 'sample'):  # already observed
            assert 0, self._state[action]
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
    
    def optimal_paths(self, state=None):
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
                if np.abs(q - best_q) < 0.01:
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

    @lru_cache(CACHE_SIZE)
    @contract
    def vpi(self, state) -> 'float, >= -0.001':
        obs = self.subtree[0]
        return (self.node_value_after_observe(obs, 0, state).expectation()
                - self.expected_term_reward(state)
                )

    def unclicked(self, state):
        return sum(1 for x in state if hasattr(x, 'sample'))
    
    @lru_cache(None) 
    def _relevant_subtree(self, node):
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

    @memoize
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
        return tuple(map(tuple, T))

    def _render(self, mode='notebook', close=False):
        if close:
            return
        from graphviz import Digraph
        from IPython.display import display
        import matplotlib as mpl
        from matplotlib.colors import rgb2hex
        
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

    def to_obs_tree(self, state, node, obs=(), sort=True):
        maybe_sort = sorted if sort else lambda x: x
        def rec(n):
            subjective_reward = state[n] if n in obs else expectation(state[n])
            children = tuple(maybe_sort(rec(c) for c in self.tree[n]))
            return (subjective_reward, children)
        # return obs_rec(self.tree, state, obs, node)
        return rec(node)

    @lru_cache(CACHE_SIZE)
    def to_obs_flat(self, state, node, obs=(), sort=False):
        s = [expectation(x) for x in state]
        for n in obs:
            s[n] = state[n]
        return tuple(s)

def flat_hash_key(args, kwargs):
    obs, node, state, tree = args
    pass

def sort_tree(env, state):
    """Breaks symmetry between belief states.
    
    This is done by enforcing that the knowldge about states at each
    depth be sorted by [0, 1, UNKNOWN]
    """
    state = list(state)
    for i in range(len(env.tree) - 1, -1, -1):
        if not env.tree[i]:
            continue
        c1, c2 = env.tree[i]
        idx1, idx2 = env.subtree_slices[c1], env.subtree_slices[c2]
        
        if not (state[idx1] <= state[idx2]):
            state[idx1], state[idx2] = state[idx2], state[idx1]
    return tuple(state)

@lru_cache(SMALL_CACHE_SIZE)
def flat_node_value_after_observe(obs_flat):
    if len(obs_flat) == 1:
        return ZERO    
    c1 = 1
    c2 = len(obs_flat) // 2 + 1
    return smax((flat_node_value_after_observe(obs_flat[c1:c2]) + obs_flat[c1],
                 flat_node_value_after_observe(obs_flat[c2:]) + obs_flat[c2]))    

@lru_cache(SMALL_CACHE_SIZE)
def node_value_after_observe(obs_tree):
    """A distribution over the expected value of node, after making an observation.
    
    `obs` can be a single node, a list of nodes, or 'all'
    """
    children = tuple(node_value_after_observe(c) + c[0] for c in obs_tree[1])
    return smax(children, default=ZERO)



@lru_cache(None)
def exact_node_value_after_observe(obs_tree, indent=''):
    """A distribution over the expected value of node, after making an observation.
    
    `obs` can be a single node, a list of nodes, or 'all'
    """
    # print(f'{indent}{obs_tree}')
    children = tuple(exact_node_value_after_observe(c, indent+'   ') + c[0]
                     for c in obs_tree[1])
    # print(f'{indent}{children}')
    x = cmax(children, default=ZERO)
    # print(f'{indent}{x}')
    return x


@lru_cache(None)
def exact_flat_node_value_after_observe(obs_flat):
    if len(obs_flat) == 1:
        return ZERO    
    c1 = 1
    c2 = len(obs_flat) // 2 + 1
    return cmax((exact_flat_node_value_after_observe(obs_flat[c1:c2]) + obs_flat[c1],
                 exact_flat_node_value_after_observe(obs_flat[c2:]) + obs_flat[c2]))

# @lru_cache(None)
# def exact_flat_node_value_after_observe(obs_flat):
#     if len(obs_flat) == 1:
#         return ZERO    
#     c1 = 1
#     c2 = len(obs_flat) // 2 + 1
#     return cmax((exact_flat_node_value_after_observe(obs_flat[c1:c2]) + obs_flat[c1],
#                  exact_flat_node_value_after_observe(obs_flat[c2:]) + obs_flat[c2]))


def obs_rec(tree, state, obs, n):
    subjective_reward = state[n] if n in obs else expectation(state[n])
    children = tuple(obs_rec(tree, state, obs, c) for c in tree[n])
    return (subjective_reward, children)

@memoize(key=lambda args, kwargs: len(args[0]))
def tree_max(obs_flat):
    c1 = 1
    c2 = len(obs_flat) // 2 + 1
    return smax((flat_node_value_after_observe(obs_flat[c1:c2]) + obs_flat[c1],
                 flat_node_value_after_observe(obs_flat[c2:]) + obs_flat[c2]))    
