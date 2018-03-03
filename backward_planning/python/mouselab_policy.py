import numpy as np
from policies import SoftmaxPolicy
from collections import OrderedDict

class MouselabPolicy(SoftmaxPolicy):
    """A linear softmax policy for MouselabEnv."""
    def __init__(self, weights, **kwargs):
        super().__init__(**kwargs)
        self.weights = OrderedDict(
            is_term=0,
            term_reward=1,
            voi_myopic=0,
            vpi_action=0,
            vpi_full=0,
            quality_ev=0,
            quality_std=0,
            depth=0,
        )
        for k in weights:
            if k not in self.weights:
                raise ValueError(f'No paramter named "{k}"')
        self.weights.update(weights)
        self.theta = np.array([*self.weights.values(), 1])  # dummy coefficient

    def preference(self, state, action):
        """Softmax is over this value."""
        # if self.env.node_quality(action, state) <= self.pruning_threshold:
        #     return - 1e9
        # if self.env.expected_term_reward(state) >= self.satisficing_threshold:
        #     return 1e9 if action == self.env.term_action else -1e9
        return np.dot(self.theta, self.phi(state, action))

    def phi(self, state, action, compute_all=False):
        env, theta = self.env, self.theta
        x = np.zeros(len(theta))
        if action == env.term_action:
            x[0] = 1
            x[1] = env.expected_term_reward(state)
            # if etr > self.satisficing_threshold:
            #     x[8] = 1e100
            return x
        else:
            if not hasattr(state[action], 'sample'):
                # already clicked this node
                x[8] = -1e100
                return x
            # Value of information
            # the `self.theta[i] and` trick skips computing if feature won't be used
            x[2] = (compute_all or theta[2]) and env.myopic_voc(action, state)
            x[3] = (compute_all or theta[3]) and env.vpi_action(action, state)
            x[4] = (compute_all or theta[4]) and env.vpi(state)

            # Value of best path through the node (given current knowledge)
            if compute_all or any(theta[5:7]):
                quality = env.node_quality(action, state)
                x[5] = quality.expectation()
                x[6] = quality.std()
        
            # Structural
            x[7] = len(env.path_to(action)) - 1 # depth
            # TODO: same_branch_as_last

        return x
