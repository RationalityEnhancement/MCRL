from collections import namedtuple, defaultdict, deque, Counter
import numpy as np
import gym
from gym import spaces
from distributions import smax, cmax, sample, expectation, Normal, PointMass, Categorical
from toolz import memoize, get
import random
from contracts import contract
from functools import reduce 
import math
import itertools as it

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

    def __init__(self, tree, init, env_type, ground_truth=None, cost=0, sample_term_reward=False,repeat_cost=1):
        
        self.init = (0, *init[1:])
        if ground_truth is not None:
            if len(ground_truth) != len(init):
                raise ValueError('len(ground_truth) != len(init)')
            self.ground_truth = np.array(ground_truth)
        else:
            self.ground_truth = np.array(list(map(sample, init)))
        self.ground_truth[0] = 0.
        self.cost = - abs(cost)
        self.sample_term_reward = sample_term_reward
        self.term_action = len(self.init)

        # Required for gym.Env API.
        self.action_space = spaces.Discrete(len(self.init) + 1)
        #self.observation_space = spaces.Box(-np.inf, np.inf, shape=len(self.init))

        self.initial_states = None  # TODO
        self.exact = True  # TODO
        self.tree = tree
        self.subtree = self._get_subtree()
        self.subtree_slices = self._get_subtree_slices()
        self.paths = self.get_paths(0)
        self._reset()
        self.repeat_cost= - abs(repeat_cost)
        self.env_type = env_type

    def _reset(self):
        if self.initial_states:
            self.init = random.choice(self.initial_states)
        self._state = self.init
        return self._state

    def _step(self, action):
        if(self.env_type=='same'):
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

        else:
            if self._state is self.term_state:
                assert 0, 'state is terminal'
            if action == self.term_action:
                reward = self._term_reward()
                done = True
                obs=False
            elif not hasattr(self._state[action], 'sample'):  # already observed
                reward = self.repeat_cost
                done = False
                obs=True
            else:  # observe a new node
                self._state = self._observe(action)
                reward = self.cost
                done = False
                obs=False
            return self._state, reward, done, obs


    def _term_reward(self):
        returns = [self.ground_truth[list(path)].sum() 
                   for path in self.optimal_paths()]
        if self.sample_term_reward:
            return np.random.sample(returns)
        else:
            return np.mean(returns)


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
        """
        Calculates myopic voc


        Using literal definition of myopic voc. Take the expectation of best possible action after observing the node
        """

        # print("Myopic VOC Observe Value = {}".format(self.node_value_after_observe((action,), 0, state).expectation()))
        possible = list(state[action])
        r,p = zip(*possible)
        expected_reward = 0
        for k in range(len(p)):
        	state2 = list(state)
        	state2[action] = r[k]
        	expected_reward +=p[k]*self.expected_term_reward(tuple(state2))
        # print(state2)
        # state2[action] = self.ground_truth[action]
        # print(state2)
        # print(state)
        # print(expected_reward, self.expected_term_reward(state))
        return(float(expected_reward - self.expected_term_reward(state)))
        # return (float(self.node_value_after_observe((action,), 0, state).expectation()
        #         - self.expected_term_reward(state))
        #         )

    # @lru_cache(CACHE_SIZE)
    @contract
    def vpi_branch(self, action, state) -> 'float, >= -0.001':
        obs = self._relevant_subtree(action)
        return (float(self.node_value_after_observe(obs, 0, state).expectation()
                - self.expected_term_reward(state))
                )
    
    @contract
    def vpi_action(self, action, state,verbose=False) -> 'float, >= -0.001':
        """
        Calculates vpi action. Nodes of importance are those who are either parents or children of the node selected
        """

        print("Ground Truth = {}".format(self.ground_truth))
        print("State = {}".format(state))
        print("Action = {}".format(action))
        if(self.env_type== "new"):
            # print("Will call path_to_new")
            obs = (*self.subtree[action][0:], *self.path_to_new(action)[1:])
        else:
            obs = (*self.subtree[action][0:], *self.path_to(action)[1:])
        print(obs)
        print("Env.state = {}".format(state))
        # for _,i in enumerate(state):
        #     print(i)
        # print("Expected Term Reward = {}".format(self.expected_term_reward(state)))
        # print("Observe Node Expected = {}".format(self.node_value_after_observe(obs, 0, state,verbose).expectation()))
        return (float(self.node_value_after_observe(obs, 0, state,verbose).expectation()
                - self.expected_term_reward(state))
                )

    @lru_cache(CACHE_SIZE)
    @contract
    def vpi(self, state) -> 'float, >= -0.001':
        """
        Calculates vpi. All nodes are important. Basically calculating vpi_action with 0th (root) node selected
        """

        obs = self.subtree[0]
        # print("Obs = {}".format(obs))
        nvao = float(self.node_value_after_observe(obs, 0, state).expectation())
        # print("VPI Node observe value = {}".format(nvao))
        return (nvao
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
    def leaves(self, node):
        trees = [self.subtree[n1] for n1 in self.tree[0]]
        for t in trees:
            if node in t:
                return tuple(t)
        assert False

    def node_value_after_observe(self, obs, node, state,verbose=False):
        """A distribution over the expected value of node, after making an observation.
        
        obs can be a single node, a list of nodes, or 'all'
        """
        obs_tree = self.to_obs_tree(state, node, obs)
        if(verbose):
            print("Obtained OBS TREE. In node_value_after_observe. obs_tree = {}\n".format(obs_tree))
        if self.exact:
            if self.env_type == "new":
        	    return exact_node_value_after_observe_new(obs_tree,verbose)
            else:
                return exact_node_value_after_observe(obs_tree)
        else:
            print("ERROR")
            return node_value_after_observe(obs_tree)

    @memoize
    def path_to_new(self,node, start=0):
        nodes = self.tree[0:3]
        for depth,nd in enumerate(nodes):
            if(node in nd):
                break
        # print("Node {} is in depth: {}".format(node,depth+1))
        path = [start]
        if (depth == 0):
            # path.append(node)
            return path
        if (depth == 1):
            for _,i in enumerate(nodes[depth-1]):
                path.append(i)
                # path.append(node)
            return path
        if (depth == 2):
            for _,i in enumerate(nodes[depth-2]):
                for _,j in enumerate(nodes[depth-1]):
                    path.append(i)
                    path.append(j)
                    # path.append(node)
            return path
        assert False

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

    @classmethod
    def new_symmetric_A(cls, branching, reward_list, seed=None, **kwargs):
        """Returns a MouselabEnv with a symmetric structure.

        Old implementation using single parent model and mapping to form a multi parent model.

        Arguments:
            branching: a list that specifies the branching factor at each depth.
        """

        if seed is not None:
            np.random.seed(seed)


        flatten_reward =[item for sublist in reward_list for item in sublist]
        reward_arr=np.array(list(map(sample, flatten_reward)))
        reward_arr = reward_arr.reshape((3,3))

        ground_truth = [0,
        reward_arr[0][0],reward_arr[1][0],reward_arr[2][0],reward_arr[2][1],reward_arr[2][2], reward_arr[1][1],reward_arr[2][0],reward_arr[2][1],reward_arr[2][2],reward_arr[1][2],reward_arr[2][0],reward_arr[2][1],reward_arr[2][2],
        reward_arr[0][1],reward_arr[1][0],reward_arr[2][0],reward_arr[2][1],reward_arr[2][2], reward_arr[1][1],reward_arr[2][0],reward_arr[2][1],reward_arr[2][2],reward_arr[1][2],reward_arr[2][0],reward_arr[2][1],reward_arr[2][2],
        reward_arr[0][2],reward_arr[1][0],reward_arr[2][0],reward_arr[2][1],reward_arr[2][2], reward_arr[1][1],reward_arr[2][0],reward_arr[2][1],reward_arr[2][2],reward_arr[1][2],reward_arr[2][0],reward_arr[2][1],reward_arr[2][2],
        ]

        reward_arr=reward_list
        init_state = [0,
        reward_arr[0][0],reward_arr[1][0],reward_arr[2][0],reward_arr[2][1],reward_arr[2][2], reward_arr[1][1],reward_arr[2][0],reward_arr[2][1],reward_arr[2][2],reward_arr[1][2],reward_arr[2][0],reward_arr[2][1],reward_arr[2][2],
        reward_arr[0][1],reward_arr[1][0],reward_arr[2][0],reward_arr[2][1],reward_arr[2][2], reward_arr[1][1],reward_arr[2][0],reward_arr[2][1],reward_arr[2][2],reward_arr[1][2],reward_arr[2][0],reward_arr[2][1],reward_arr[2][2],
        reward_arr[0][2],reward_arr[1][0],reward_arr[2][0],reward_arr[2][1],reward_arr[2][2], reward_arr[1][1],reward_arr[2][0],reward_arr[2][1],reward_arr[2][2],reward_arr[1][2],reward_arr[2][0],reward_arr[2][1],reward_arr[2][2],
        ]


        def reward(depth):
            if depth == 1:
                return 1
            elif depth == 2:
                return 2
            elif depth == 3:
                return 3
            else:
                return 0


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
        return cls(tree, init_state,ground_truth=ground_truth, **kwargs)


    @classmethod
    def new_symmetric(cls, branching, reward, seed=None, env_type="new", initval = None, **kwargs):
        """Returns a MouselabEnv with a symmetric structure.
        
        Arguments:
            branching: a list that specifies the branching factor at each depth.
            reward: a function that returns the reward distribution at a given depth.
        """
        
        if seed is not None:
            np.random.seed(seed)
        if not callable(reward):
            r = reward
            reward = lambda depth: r

        init = []
        tree = []

        def expand(d):
            """
            Create the structure of the environment where a child has single parent

            Tree consists of a tuple stating the children of node of index. Ini is tuple of the  reward distribution of nodes
            """

            my_idx = len(init)
            init.append(reward(d))
            children = []
            tree.append(children)
            for _ in range(get(d, branching, 0)):
                child_idx = expand(d+1)
                children.append(child_idx)
            return my_idx

        # NEW CODE START

        branching1 = branching.copy()
        branching1.append(0)
        init = []
        tree = []
        MAX_SIZE = reduce((lambda x, y: x + y), branching) # Add all terms in a list -> Max number of nodes for new MDP
        flag = [0] * (MAX_SIZE)
        node_idx = []
        MAX = 0
        #maxs = []
        for i in range(len(branching)):
            node_idx.extend(list(range(MAX+1, MAX+branching[i]+1)))
            MAX = max(node_idx)
            #maxs.append(MAX)

        def expand_new(d):
            """
            Create the structure of the environment where a child has multiple parents like a fully connected neural network. 

            Tree consists of a tuple stating the children of node of index. Ini is tuple of the  reward distribution of nodes
            """

        #   print(d)
            children = []
            if(d!=0 and d <=3):
        #         print("For layer {}, the node index can be:  {}".format(d,[d]+list(range(sum(branching[d:])+d+1, sum(branching[d-1:])+d))))
                tt = [flag[i-1] for i in[d]+list(range(sum(branching[d:])+d+1, sum(branching[d-1:])+d))]
                node_t = [node_idx[i-1] for i in[d]+list(range(sum(branching[d:])+d+1, sum(branching[d-1:])+d))]
                if(sum(tt)< branching[d-1]):
                    my_idx = node_t[sum(tt)]
                else: #Atleast one initiation of node exist
                    my_idx = node_t[tt.index(min(tt))]
        #             print("Index to prevent overflow: {}".format(my_idx))
        #         print("Index: {}".format(my_idx))
                if(flag[my_idx-1] == 0):
                    init.append(reward(d))
                    tree.append(children)
        #             print("Init Lenght: {}".format(len(init)))
                flag[my_idx-1] += 1
            else:
                my_idx = len(init)
        #         print("Index for d=0 or > 3: {}".format(my_idx))
            if(d==0):
                init.append(reward(d))
                tree.append(children)
            for _ in range(get(d, branching1, 0)):
        #         print("Node {} Calling".format(my_idx))
                child_idx = expand_new(d+1)
                children.append(child_idx)
        #     print("Returning Index: {}".format(my_idx))
            return my_idx



        if(env_type == "new"):
            expand_new(0)
        else:
            expand(0)

        # expand(0) ## UNCOMMENT FOR ORIGINAL
        if initval is not None:
            init = initval
        # NEW CODE END
        return cls(tree, init, env_type, **kwargs)

    def _render(self, mode='notebook', close=False):
        """
        Renders the environment structute
        """

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
        """
        Takes the number of nodes that can be observed and creates multi-nested tuple for each possible path
        Nodes which aren't observable are set to its expected value (normally 0)
        """

        # print("IN TO OBS TREE")
        maybe_sort = sorted if sort else lambda x: x
        def rec(n):
            # print("In rec({})".format(n))
            subjective_reward = state[n] if n in obs else expectation(state[n])
            # print("state[n] = {}".format(state[n]))
            # print("Subjective Reward = {}, n in obs: {} state[n] = {}, expectation(state[n]) = {}".format(subjective_reward, n in obs, state[n], expectation(state[n])))
            # print("n = {} self.tree[n] = {}".format(n, self.tree[n]))
            children = tuple(maybe_sort(rec(c) for c in self.tree[n]))
            # print("Children = {}".format(children))
            # print("Returning for n={} RETURNED = {}".format(n,(subjective_reward, children)))
            return (subjective_reward, children)
        return rec(node)

@lru_cache(SMALL_CACHE_SIZE)
def node_value_after_observe(obs_tree):
    """A distribution over the expected value of node, after making an observation.
    
    `obs` can be a single node, a list of nodes, or 'all'
    """

    children = tuple(node_value_after_observe(c) + c[0] for c in obs_tree[1])
    return smax(children, default=ZERO)

def exact_node_value_after_observe(obs_tree):
    """A distribution over the expected value of node, after making an observation.

    `obs` can be a single node, a list of nodes, or 'all'
    """

    # print("IN EXACT NODE VALUE AFTER Observe\n")
    # print("OT = {}".format(obs_tree))

    children = tuple(exact_node_value_after_observe(c) + c[0]
                     for c in obs_tree[1])
    val = cmax(children, default=ZERO)
    # print("For obs_tree = {}, children={}".format(obs_tree, children))
    # print("val = {}".format(val))
    return val
    return cmax(children, default=ZERO)

def exact_node_value_after_observe_new(obs_tree,verbose=False):
    """A distribution over the expected value of node, after making an observation.

    `obs` can be a single node, a list of nodes, or 'all'

    Since a child can have multiple parents, the parents distributions are joint to make a pseudo parent for easy computation
    """

    # print("IN EXACT NODE VALUE AFTER Observe NEW\n")
    # print("OT = {}".format(obs_tree))

    if(verbose):
        print("obs_tree = {}, length = {},type = {}, type1={}".format(obs_tree, len(obs_tree[1]), type(obs_tree), type(obs_tree[1])))
    MC = np.NINF
    flag1 = False # Sanity flag
    # flag2 = False # TRUE: there is a categorical structure that has bigger value. Need to explore
    immediate_children = []
    for c in obs_tree[1]:
        immediate_children.append(c[0])
        if(verbose):
            print("c[0] = {}".format((c[0])))
        if(c[0] > MC):
            MC = c[0]
            CC = c
            flag1 = True
            if(verbose):
                print(MC, type(CC), type(c))
    if(flag1):
        rew = []
        pro = []
        # print("Flag1 = {}, MC = {}".format(flag1, MC))
        if(verbose):
            print("LOOK {}".format(immediate_children))
        for i,child in enumerate(immediate_children):
            if(hasattr(child, 'sample')):
                r,p = zip(*child)
            else:
                r = [child]
                p = [1]
            rew.append(list(r))
            pro.append(list(p))
        reward_list = list(it.product(*rew))
        prob_list = list(it.product(*pro))
        if(verbose):
            print("Reward List = {} len = {}, Prob_list = {}".format(reward_list,len(reward_list),prob_list))
        if(len(reward_list) > 1):
            outcomes = Counter()
            for i,r in enumerate(reward_list):
                p = prob_list[i]
                outcomes[max(*r)] += reduce(lambda x, y: x*y, p)
            if(verbose):
                print("LOOK1 {}".format(outcomes))
            obs_tree = (obs_tree[0],(((Categorical(outcomes.keys(), outcomes.values()),CC[1]),)))
    
    children = tuple(exact_node_value_after_observe_new(c,verbose) + c[0]
               for c in obs_tree[1])
    val = cmax(children,verbose,default=ZERO)
        
    if(verbose):
        print("For obs_tree = {}, children={}".format(obs_tree, children))
        
        print("val = {}".format(val))
    return val

   
