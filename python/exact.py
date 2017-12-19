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
        idx1, idx2 = env.subtree_slices[c1], env.subtree_slices[c2]
        
        if not (state[idx1] <= state[idx2]):
            state[idx1], state[idx2] = state[idx2], state[idx1]
    return tuple(state)


def hash_tree(env, state):
    """Breaks symmetry between belief states.
    
    This is done by enforcing that the knowldge about states at each
    depth be sorted by [0, 1, UNKNOWN]
    """
    if state == '__term_state__':
        return hash(state)
    def rec(n):
        x = hash(state[n])
        childs =  sum(rec(c) for c in env.tree[n])
        return hash(str(x + childs))

    return rec(0)

def solve(env, hash_state=None, actions=None, blinkered=None):
    """Returns Q, V, pi, and computation data for an mdp environment."""
    if hasattr(env, 'n_arm'):
        hash_state = lambda state: tuple(sorted(state))
    elif hasattr(env, 'tree'):
        # hash_state = lambda state: sort_tree(env, state)
        hash_state = lambda state: hash_tree(env, state)
    if actions is None:
        actions = env.actions
    if blinkered == 'recursive':
        def subset_actions(a):
            if a == env.term_action:
                return ()
            return (*env.subtree[a][1:], *env.path_to(a)[:-1], env.term_action)
    elif blinkered == 'children':
        def subset_actions(a):
            if a == env.term_action:
                return ()
            return (*env.subtree[a][1:], env.term_action)
    elif blinkered == 'branch':
        assert hasattr(env, '_relevant_subtree')
        def subset_actions(a):
            if a == env.term_action:
                return ()
            else:
                return (*env._relevant_subtree(a), env.term_action)
    elif blinkered:
        def subset_actions(a):
            return (a, env.term_action)
        
    else:
        subset_actions = lambda a: None



    info = {  # track number of times each function is called
        'q': 0,
        'v': 0
    }
    if hash_state is not None:
        def hash_key(args, kwargs):
            state = args[0]
            if state is None:
                return state
            else:
                if kwargs:
                    # Blinkered approximation. Hash key is insensitive
                    # to states that can't be acted on, except for the
                    # best expected value
                    # Embed the action subset into the state.
                    action_subset = kwargs['action_subset']
                    mask = [0] * len(state)
                    for a in action_subset:
                        mask[a] = 1
                    state = tuple(zip(state, mask))
                return hash_state(state)
    else:
        hash_key = None

    @memoize
    def Q(s, a):
        # print('Q', s, a)
        info['q'] += 1
        action_subset = subset_actions(a)
        return sum(p * (r + V(s1, action_subset)) for p, s1, r in env.results(s, a))

    @memoize(key=hash_key)
    def V(s, action_subset=None):
        if s is None:
            return 0
        info['v'] += 1
        acts = actions(s)
        if action_subset is not None:
            acts = tuple(a for a in acts if a in action_subset)
        return max((Q(s, a) for a in acts), default=0)
    
    @memoize
    def pi(s):
        return max(actions(s), key=lambda a: Q(s, a))
    
    return Q, V, pi, info



