import numpy as np
from toolz import memoize

from agents import Agent
from mouselab import MouselabEnv
from distributions import Normal, Categorical
from policies import FixedPlanPolicy

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
        idx1, idx2 = env.subtrees[c1], env.subtrees[c2]
        
        if not (state[idx1] <= state[idx2]):
            state[idx1], state[idx2] = state[idx2], state[idx1]
    return tuple(state)


def solve(env, hash_state=None, actions=None):
    """Returns Q, V, pi, and computation data for an mdp environment."""
    if hash_state is None:
        hash_state = lambda state: sort_tree(env, state)
    if actions is None:
        actions = env.actions
    info = {  # track number of times each function is called
        'q': 0,
        'v': 0
    }
    
    def hash_key(args, kwargs):
        s = args[0]
        if s is None:
            return s
        else:
            return hash_state(args[0])

    @memoize
    def Q(s, a):
        info['q'] += 1
        return sum(p * (r + V(s1)) for p, s1, r in env.results(s, a))

    @memoize(key=hash_key)
    def V(s):
        if s is None:
            return 0
        info['v'] += 1
        return max((Q(s, a) for a in actions(s)), default=0)
    
    @memoize
    def pi(s):
        return max(actions(s), key=lambda a: Q(s, a))
    
    return Q, V, pi, info