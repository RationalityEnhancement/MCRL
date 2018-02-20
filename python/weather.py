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


class TornadoEnv(gym.Env):
    """Decide whether or not to evacuate each city."""
    term_state = '__term_state__'

    def __init__(self, n_city, evac_cost, false_neg_cost, sim_cost=0, max_sims=10, prior=(.1, .9)):
        super().__init__()
        self.n_city = self.n_arm = n_city
        self.evac_cost = -abs(evac_cost)
        self.false_neg_cost = -abs(false_neg_cost)
        self.sim_cost = -abs(sim_cost)
        self.max_sims = max_sims
        self.prior = prior

        self.init = (prior,) * n_city
        self._cities = range(n_city)
        self._actions = range(n_city + 1)
        self.term_action = self._actions[-1]
        self._max_sum = max_sims + sum(concat(self.init))
        self._reset()
        self.n_actions = n_city + 1

    def __hash__(self):
        return hash((self.n_city, self.evac_cost, self.false_neg_cost,
                     self.sim_cost, self.max_sims, self.prior))

    def _reset(self):
        self._state = self.init
        return self._state

    def _step(self, action):
        if action == self.term_action:
            return self.term_state, self.expected_term_reward(self._state), True, {}
        else:
            city = action
            p = self.p_hit(self._state, city)
            
            s = list(self._state)
            a, b = s[city]
            if np.random.rand() < p:
                s[city] = (a + 1, b)
            else:
                s[city] = (a, b + 1)
            self._state = tuple(s)

            return self._state, self.sim_cost, False, {}


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
            self.sim_cost,
            self.myopic_voc(state, action) - etr,
            self.vpi_action(state, action) - etr,
            self.vpi(state) - etr,
            etr
        ])

    def p_hit(self, state, city):
        a, b = state[city]
        return a / (a + b)

    def value(self, state, city):
        a, b = state[city]
        p = a / (a + b)
        return max(p * self.false_neg_cost, self.evac_cost)

    def results(self, state, action):
        if action == self.term_action:
            yield (1, self.term_state, self.expected_term_reward(state))
        else:
            city = action
            p = self.p_hit(state, city)

            a, b = state[city]
            s = list(state)
            s[city] = (a + 1, b)
            yield (p, tuple(s), self.sim_cost)
            s[city] = (a, b + 1)
            yield (1 - p, tuple(s), self.sim_cost)
    
    @memoize(key=memo_key)
    def expected_term_reward(self, state):
        return sum(self.value(state, a) for a in self._cities)
    
    @memoize(key=memo_key)
    def myopic_voc(self, state, action):
        val = sum(p * self.expected_term_reward(s1)
                  for p, s1, r in self.results(state, action))
        return val 
    
    @memoize(key=memo_key)
    def vpi(self, state):
        return sum(self.vpi_action(state, a) for a in self.actions(state))    
    
    @memoize(key=memo_key)
    def vpi_action(self, state, action):
        a, b = state[action]
        fnc = self.false_neg_cost
        return fnc * expected_min_beta_constant(a, b, self.evac_cost / fnc)
    
    # @memoize(key=memo_key)
    # def vpi_action(self, state, action):
    #     value = lambda i: state[i][0] / (state[i][0] + state[i][1])
    #     best_arm = max(self._cities, key=value)
    #     city
    #     if action == best_arm:
    #         others = (act for act in self._cities if act != action)
    #         competing_value = max(map(value, others))
    #     else:
    #         competing_value = value(best_arm)
    #     a, b = state[action]
    #     val = np.maximum(beta_samples(a, b, 0), competing_value).mean()
    #     return val - self.expected_term_reward(state)


@lru_cache(None)
def expected_min_beta_constant(a, b, c):
    return np.minimum(_beta_samples(a, b), c).mean()

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
