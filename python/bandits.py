from toolz import concat, memoize, reduce
from functools import lru_cache
from scipy.stats import beta
import numpy as np
import gym


N_SAMPLE = 10000

def memo_key(args, kwargs):
    """Symmetry breaking hashing."""
    env = args[0]
    state = args[1]
    if len(args) > 2:
        action = args[2]
        mask = [0] * len(state)
        mask[action] = 1
        state = zip(state, mask)
        state = ((s, i == action) for i, s in enumerate(state))
    return sum(map(hash, state))
    # return (env, tuple(sorted(state)))




class MetaBanditEnv(gym.Env):
    """Metalevel Bernoulli problem."""
    term_state = '__term_state__'

    def __init__(self, n_arm=2, max_obs=10, cost=0, constant=-1):
        super().__init__()
        self.n_arm = n_arm
        self.cost = -abs(cost)
        self.init = ((1, 1),) * n_arm
        self.constant = constant
        self._arms = range(n_arm)
        self._actions = range(n_arm + 1)
        self.term_action = self._actions[-1]
        self._max_sum = max_obs + sum(concat(self.init))
        self._reset()
        self.n_actions = n_arm + 1

    def _reset(self):
        self._state = self.init
        return self._state

    def _step(self, action):
        if action == self.term_action:
            return self.term_state, self.expected_term_reward(self._state), True, {}
        else:
            arm = action
            p = self.p_win(self._state, arm)
            
            s = list(self._state)
            a, b = s[arm]
            if np.random.rand() < p:
                s[arm] = (a + 1, b)
            else:
                s[arm] = (a, b + 1)
            self._state = tuple(s)

            return self._state, self.cost, False, {}


    def actions(self, state):
        if state is self.term_state:
            return []
        elif sum(concat(state)) == self._max_sum:
            return [self.term_action]
        else:
            return self._actions

    def action_features(self, action, state=None):
        state = state if state is not None else self._state
        if action == self.term_action:
            return np.array([
                0,
                0,
                0,
                0,
                self.expected_term_reward(state)
            ])
        etr = self.expected_term_reward(state)
        return np.array([
            self.cost,
            self.myopic_voc(state, action) - etr,
            self.vpi_action(state, action) - etr,
            self.vpi(state) - etr,
            etr
        ])

    def p_win(self, state, arm):
        a, b = state[arm]
        return a / (a + b)

    def value(self, state, arm):
        a, b = state[arm]
        p = a / (a + b)
        return p

    def results(self, state, action):
        if action == self.term_action:
            yield (1, self.term_state, self.expected_term_reward(state))
        else:
            arm = action
            p = self.p_win(state, arm)

            a, b = state[arm]
            s = list(state)
            s[arm] = (a + 1, b)
            yield (p, tuple(s), self.cost)
            s[arm] = (a, b + 1)
            yield (1 - p, tuple(s), self.cost)
    
    @memoize(key=memo_key)
    def expected_term_reward(self, state):
        best_value = max(self.value(state, a) for a in self._arms)
        return max(best_value, self.constant)
    
    @memoize(key=memo_key)
    def myopic_voc(self, state, action):
        val = sum(p * self.expected_term_reward(s1)
                  for p, s1, r in self.results(state, action))
        return val 
    
    @memoize(key=memo_key)
    def vpi(self, state):
        abis = tuple((a, b, i) for i, (a, b) in enumerate(state))
        val = expected_max_betas(abis)
        # samples = (beta_samples(a,b,i) for i, (a, b) in enumerate(state))
        # val = reduce(np.maximum, samples).mean()
        return val 
    
    @memoize(key=memo_key)
    def vpi_action(self, state, action):
        val_acts = sorted((self.p_win(state, a), a) for a in self._arms)
        best_arm = val_acts[0][1]

        if action == best_arm:
            competing_value = val_acts[1][0]
        else:
            competing_value = val_acts[0][0]
        a, b = state[action]
        val = expected_max_beta_constant(a, b, competing_value)
        return val 
    
    # @memoize(key=memo_key)
    # def vpi_action(self, state, action):
    #     value = lambda i: state[i][0] / (state[i][0] + state[i][1])
    #     best_arm = max(self._arms, key=value)
    #     arm
    #     if action == best_arm:
    #         others = (act for act in self._arms if act != action)
    #         competing_value = max(map(value, others))
    #     else:
    #         competing_value = value(best_arm)
    #     a, b = state[action]
    #     val = np.maximum(beta_samples(a, b, 0), competing_value).mean()
    #     return val - self.expected_term_reward(state)


@lru_cache(None)
def expected_max_betas(abis):
    return max_betas(abis).mean()

@lru_cache(None)
def expected_max_beta_constant(a, b, constant):
    return np.maximum(_beta_samples(a, b), constant).mean()

@lru_cache(None)
def max_betas(abis):
    if len(abis) == 1:
        return beta_samples(*abis[0])
    else:
        # Divide and conquer!
        split = len(abis) // 2
        return np.maximum(max_betas(abis[:split]),
                          max_betas(abis[split:]))

@lru_cache(None)
def _beta_samples(a, b):
    return beta(a, b).rvs(N_SAMPLE)

@lru_cache(None)
def beta_samples(a, b, idx):
    samples = _beta_samples(a, b).copy()
    np.random.shuffle(samples)
    return samples
