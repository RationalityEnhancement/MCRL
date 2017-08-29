from toolz import concat

class MetaBanditEnv(object):
    """Metalevel Bernoulli problem."""
    term_state = 'term_state'

    def __init__(self, n_arm=2, max_obs=10, cost=0):
        super().__init__()
        self.n_arm = n_arm
        self.cost = -abs(cost)
        self.init = ((1, 1),) * n_arm
        self._arms = range(n_arm)
        self._actions = range(n_arm + 1)
        self.term_action = self._actions[-1]
        self._max_sum = max_obs + sum(concat(self.init))

    def actions(self, state):
        if state is self.term_state:
            return []
        elif sum(concat(state)) == self._max_sum:
            return [self.term_action]
        else:
            return self._actions

    # def get_idx(self, arm):
    #     return slice(arm*2, arm*2 + 2)

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

    def expected_term_reward(self, state):
        return max(self.p_win(state, a) for a in self._arms)




