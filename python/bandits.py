from toolz import concat, memoize, reduce
from scipy.stats import beta
import numpy as np
import gym

def memo_key(args, kwargs):
    """Symmetry breaking hashing."""
    state = args[1]
    return (tuple(sorted(state)), *args[1:])



class MetaBanditEnv(gym.Env):
    """Metalevel Bernoulli problem."""
    term_state = 'term_state'

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

        return np.array([
            self.cost,
            self.myopic_voc(state, action),
            self.vpi_action(state, action),
            self.vpi(state),
            self.expected_term_reward(state)
        ])

    def p_win(self, state, arm):
        a, b = state[arm]
        return a / (a + b)

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
        best_value = max(self.p_win(state, a) for a in self._arms)
        return max(best_value, self.constant)
    
    @memoize(key=memo_key)
    def myopic_voc(self, state, action):
        val = sum(p * self.expected_term_reward(s1)
                  for p, s1, r in self.results(state, action))
        return val - self.expected_term_reward(state)
    
    @memoize(key=memo_key)
    def vpi(self, state):
        samples = (beta_samples(a,b) for a, b in state)
        val = reduce(np.maximum, samples).mean()
        return val - self.expected_term_reward(state)
    
    @memoize(key=memo_key)
    def vpi_action(self, state, action):
        def value(act):
            return self.p_win(state, act)
        best_arm = max(self._arms, key=value)
        if action == best_arm:
            others = (act for act in self._arms if act != action)
            competing_value = max(map(value, others))
        else:
            competing_value = value(best_arm)
        a, b = state[action]
        val = np.maximum(beta_samples(a, b), competing_value).mean()
        return val - self.expected_term_reward(state)

@memoize
def _beta_samples(a, b, n):
    return beta(a, b).rvs(n)

def beta_samples(a, b, n=100000):
    samples = _beta_samples(a, b, n)
    np.random.shuffle(samples)
    return samples
