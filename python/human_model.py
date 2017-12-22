from skopt import gp_minimize
import numpy as np

from evaluation import get_util
from policies import Policy
from contexttimer import Timer
from joblib import Parallel, delayed
from toolz import curry, concat
from utils import softmax

from policies import SoftmaxPolicy


class HumanPolicy(SoftmaxPolicy):
    """A linear softmax policy"""
    def __init__(self, theta, **kwargs):
        super().__init__(**kwargs)
        self.theta = np.array(theta)

    @curry
    def preference(self, state, action):
        """Softmax is over this value."""
        return np.dot(self.theta, self.phi(state, action))

    def phi(self, state, action):
        env, theta = self.env, self.theta
        x = np.zeros(len(theta))
        if action == env.term_action:
            x[0] = 1
            x[1] = env.expected_term_reward(state)
            return x
        else:
            if not hasattr(state[action], 'sample'):
                return x  # already clicked this node
            # Value of information
            # the `self.theta[i] and` trick skips computing if feature won't be used
            x[2] = theta[2] and env.myopic_voc(action, state)
            x[3] = theta[3] and env.vpi_action(action, state)
            x[4] = theta[4] and env.vpi(state)

            # Value of best path through the node (given current knowledge)
            if any(theta[5:7]):
                quality = env.node_quality(action, state)
                x[5] = quality.expectation()
                x[6] = quality.std()
        
            # Structural
            x[7] = len(env.path_to(action)) # depth
            # TODO: same_branch_as_last

        return x
