from toolz import memoize

def blinkered(env, hash_state=None, actions=None):
    """Returns Q, V, pi, and computation data for an mdp environment."""
    if actions is None:
        actions = env.actions

    info = {  # track number of times each function is called
        'q': 0,
        'v': 0
    }
    
    if hash_state is not None:
        def hash_key(args, kwargs):
            s = args[0]
            if s is None:
                return s
            else:
                return hash_state(args[0])
    else:
        hash_key = None

    @memoize
    def Q(s, a):
        info['q'] += 1
        actions = (a, env.term_action)
        return sum(p * (r + V(s1, actions)) for p, s1, r in env.results(s, a))

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