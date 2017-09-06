from toolz import concat, memoize, reduce
from scipy.stats import beta
import numpy as np

class MetaBanditEnv(object):
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

    def _reset(self):
        self._state = [[1, 1],] * self.n_arm
        return self._state

    def _step(self, action):
        if action == self.term_action:
            return self.term_state, self.expected_term_reward(self._state), True, {}
        else:
            arm = action
            p = self.p_win(self._state, arm)
            if np.random.rand() < p:
                self._state[arm][0] += 1
            else:
                self._state[arm][1] += 1
            return self._state, self.cost, False, {}


    def actions(self, state):
        if state is self.term_state:
            return []
        elif sum(concat(state)) == self._max_sum:
            return [self.term_action]
        else:
            return self._actions

    def action_features(self, action, state):
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

    @memoize
    def expected_term_reward(self, state):
        best_value = max(self.p_win(state, a) for a in self._arms)
        return max(best_value, self.constant)

    @memoize
    def myopic_voc(self, action, state):
        val = sum(p * self.expected_term_reward(s1)
                  for p, s1, r in self.results(state, action))
        return val - self.expected_term_reward(state)

    @memoize
    def vpi(self, state):
        samples = (beta_samples(a,b) for a, b in state)
        val = reduce(np.maximum, samples).mean()
        return val - self.expected_term_reward(state)

    @memoize
    def vpi_action(self, action, state):
        def value(act):
            return self.p_win(state, act)
        best_arm = max(self._arms, key=value)
        if action == best_arm:
            others = (act for act in self._arms if act != action)
            competing_value = max(map(value, others))
        else:
            competing_value = value(best_arm)
        a, b = state[action]
        # print('competing_value', competing_value)
        val = np.maximum(beta_samples(a, b), competing_value).mean()
        return val - self.expected_term_reward(state)

def beta_samples(a, b, n=100000):
    return beta(a, b).rvs(n)
