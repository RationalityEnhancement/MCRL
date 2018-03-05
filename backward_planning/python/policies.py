from collections import namedtuple, defaultdict, Counter, deque
import itertools as it
import numpy as np
from abc import ABC, abstractmethod
from utils import PriorityQueue
import time
np.set_printoptions(precision=3, linewidth=200)
from scipy import stats
from tqdm import tqdm, trange, tnrange
from copy import deepcopy
import random
from agents import Component, Model
from utils import softmax

class Policy(Component):
    """Chooses actions."""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def act(self, state):
        """Returns an action to take in the given state."""
        pass

    def attach(self, agent):
        if not hasattr(agent, 'env'):
            raise ValueError('Must attach env before attaching policy.')
        super().attach(agent)


class FunctionPolicy(Policy):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def act(self, state):
        return self.policy(state)

class RandomPolicy(Policy):
    """Chooses actions randomly."""
    def __init__(self):
        super().__init__()
    
    def act(self, state):
        try:
            return random.choice(list(self.env.actions(state)))
        except:
            return self.env.action_space.sample()


class SoftmaxPolicy(Policy):
    """Samples actions from a softmax over preferences."""
    def __init__(self, preference=None, temp=1e-9, noise=1e-9):
        super().__init__()
        if preference is None:
            assert hasattr(self, 'preference')
        else:
            self.preference = preference
        self.temp = temp
        self.noise = noise
    
    def act(self, state):
        probs = self.action_distribution(state)
        probs += np.random.rand(len(probs)) * self.noise
        probs /= probs.sum()
        return np.random.choice(len(probs), p=probs)

    def action_distribution(self, state):
        q = np.zeros(self.n_action) - 1e30
        for a in self.env.actions(state):
            q[a] = self.preference(state, a)
        return softmax(q, self.temp)

    def preferences(self, state):
        q = np.zeros(self.n_action) - 1e30
        for a in self.env.actions(state):
            q[a] = self.preference(state, a)
        return q


class RandomTreePolicy(Policy):
    """Chooses actions randomly."""
    def __init__(self):
        super().__init__()
    
    def act(self, state):
        actions = list(self.env.actions(self.env._state))
        return random.choice(actions)

class MaxQPolicy(Policy):
    """Chooses the action with highest Q value."""
    def __init__(self, Q, epsilon=0.5, anneal=.95, **kwargs):
        super().__init__(**kwargs)
        self.Q = Q
        self.epsilon = epsilon
        self.anneal = anneal

    def act(self, state, anneal_step=0):
        q = self.Q.predict(state)
        epsilon = self.epsilon * self.anneal ** anneal_step
        if np.random.rand() < epsilon:
            noise = np.random.random(q.shape) * 1000
        else:
            noise = np.random.random(q.shape) * .001
        return np.argmax(q + noise)


class LiederPolicy(Policy):
    """The meta-policy of Lieder et al. (2017) AAAI."""
    def __init__(self, theta, bad_max=False):
        self.theta = np.array(theta)
    
    def act(self, state):
        def Q(a):
            if a == self.env.term_action:
                return self.env.expected_term_reward(self.env._state)
            else:
                return np.dot(self.theta, self.env.action_features(a))
        action = max(self.env.actions(state), key=Q)
        return action


class MaxQSamplePolicy(Policy):
    """Chooses the action with highest sampled Q value.

    `Q.predict` must have the kwarg `return_var`."""
    def __init__(self, Q, **kwargs):
        super().__init__(**kwargs)
        self.Q = Q
        self.save_regret = True

    def act(self, state):
        q, var = self.Q.predict(state, return_var=True)
        sigma = var ** 0.5
        q_samples = q + np.random.randn() * sigma
        a = np.argmax(q_samples)
        if self.save_regret:
            q = q.flat
            a1 = np.argmax(q)
            self.save('max', a == a1)
            self.save('regret', q[a1] - q[a])
        return a


class ActorCritic(Policy):
    """docstring for ActorCritic"""
    def __init__(self, critic, actor_lr=0.001, discount=.99, actor_lambda=1, **kwargs):
        super().__init__()
        self.critic = critic
        self.discount = discount
        self.actor_lambda = actor_lambda
        self.actor_lr = actor_lr

        self._actor_discount = np.array([(self.discount * self.actor_lambda) ** i 
                                         for i in range(5000)])
    

        self.memory = deque(maxlen=100)
        self.batch_size = 20

    def attach(self, agent):
        super().attach(agent)
        self.actor = self.build_actor()

    def build_actor(self):
        from keras.layers import Dense
        from keras.models import Sequential
        from keras.optimizers import Adam, Nadam
        actor = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu',
                  kernel_initializer='he_uniform'),
            # Dense(24, activation='relu',
            #       kernel_initializer='he_uniform'),
            Dense(self.n_action, activation='softmax',
                  kernel_initializer='he_uniform')
        ])
        # actor.summary()
        actor.compile(loss='categorical_crossentropy',
                      optimizer=Nadam(self.actor_lr))
        return actor

    def act(self, state):
        policy = self.actor.predict(state.reshape(1, -1)).flatten()
        return np.random.choice(self.n_action, 1, p=policy)[0]

    # update networks every episode
    def finish_episode(self, trace):
        if len(self.memory) >= self.batch_size:
            batch = np.random.choice(self.memory, self.batch_size)
            batch.append(trace)
            self.train_batch(batch)

    def train_batch(self, traces):
        def data():
            for trace in traces:
                yield trace['states'][:-1], trace['actions'], trace['rewards']

        state, action, reward = map(np.concatenate, zip(*data()))
        n_step = len(state)
        # See Schulman et al. 2016 ICLR paper
        value = np.r_[self.critic.predict(state).flat, 0]  # final state value
        delta = reward + self.discount * value[1:] - value[:-1]  # pg. 4
        advantage = np.zeros((n_step, self.n_action))
        value_target = np.zeros((n_step, 1))

        for i in range(n_step):
            adv = np.sum(delta[i:] * self._actor_discount[:n_step-i])
            advantage[i, action[i]] = adv
            val_error = np.sum(delta[i:] * self._critic_discount[:n_step-i])
            value_target[i, 0] = value[i] + val_error
            # value_target[i, 0] = reward[i] + self.discount * value[i+1]

        self.actor.fit(state, advantage, epochs=1, verbose=0)
        self.critic.fit(state, value_target, epochs=1, verbose=0)

    def train(self, trace):
        state = np.stack(trace['states'][:-1])  # ignore final state
        action = trace['actions']
        reward = trace['rewards']
        n_step = len(state)

        # See Schulman et al. 2016 ICLR paper
        value = np.r_[self.critic.predict(state).flat, 0]  # final state value
        delta = reward + self.discount * value[1:] - value[:-1]  # pg. 4
        advantage = np.zeros((n_step, self.n_action))
        value_target = np.zeros((n_step, 1))

        for i in range(n_step):
            adv = np.sum(delta[i:] * self._actor_discount[:n_step-i])
            advantage[i, action[i]] = adv
            val_error = np.sum(delta[i:] * self._critic_discount[:n_step-i])
            value_target[i, 0] = value[i] + val_error
            # value_target[i, 0] = reward[i] + self.discount * value[i+1]

        self.actor.fit(state, advantage, epochs=1, verbose=0)
        self.critic.fit(state, value_target, epochs=1, verbose=0)


class GeneralizedAdvantageEstimation(Policy):
    """A variance-reducing extension of Advantage Actor Critic.

    https://arxiv.org/abs/1506.02438
    """
    def __init__(self, actor_lr=0.001, critic_lr= 0.005, discount=.99, actor_lambda=1, critic_lambda=1, **kwargs):
        super().__init__()
        self.discount = discount
        self.actor_lambda = actor_lambda
        self.critic_lambda = critic_lambda
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self._actor_discount = np.array([(self.discount * self.actor_lambda) ** i 
                                         for i in range(5000)])
        self._critic_discount = np.array([(self.discount * self.critic_lambda) ** i 
                                          for i in range(5000)])

        self._memory = deque(maxlen=100)
        self.batch_size = 20

    def attach(self, agent):
        super().attach(agent)
        self.actor = self.build_actor()
        self.critic = self.build_critic()

    def build_actor(self):
        from keras.layers import Dense
        from keras.models import Sequential
        from keras.optimizers import Nadam
        actor = Sequential([
            # Dense(24, input_dim=self.state_size, activation='relu',
            #       kernel_initializer='he_uniform'),
            # Dense(24, activation='relu',
            #       kernel_initializer='he_uniform'),
            Dense(self.n_action, input_dim=self.state_size, activation='softmax',
                  kernel_initializer='he_uniform')
        ])
        # actor.summary()
        actor.compile(loss='categorical_crossentropy',
                      optimizer=Nadam(self.actor_lr))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):
        from keras.layers import Dense
        from keras.models import Sequential
        from keras.optimizers import Nadam
        critic = Sequential([
            # Dense(24, input_dim=self.state_size, activation='relu',
            #       kernel_initializer='he_uniform'),
            # Dense(24, activation='relu',
            #       kernel_initializer='he_uniform'),
            Dense(1, input_dim=self.state_size, activation='linear',
                  kernel_initializer='he_uniform')
        ])
        # critic.summary()
        critic.compile(loss="mse", optimizer=Nadam(self.critic_lr))
        return critic

    def act(self, state):
        policy = self.actor.predict(state.reshape(1, -1)).flatten()
        return np.random.choice(self.n_action, 1, p=policy)[0]

    # update networks every episode
    def finish_episode(self, trace):
        if len(self._memory) >= self.batch_size:
            batch = np.random.choice(self._memory, self.batch_size)
            batch.append(trace)
            self.train_batch(batch)
        else:
            self.train(trace)

    def train_batch(self, traces):
        def data():
            for trace in traces:
                yield trace['states'][:-1], trace['actions'], trace['rewards']

        state, action, reward = map(np.concatenate, zip(*data()))
        n_step = len(state)
        # See Schulman et al. 2016 ICLR paper
        # value = np.r_[self.critic.predict(state).flat, 0]  # final state value
        value = np.r_[state.sum(1), 0]
        delta = reward + self.discount * value[1:] - value[:-1]  # pg. 4
        advantage = np.zeros((n_step, self.n_action))
        value_target = np.zeros((n_step, 1))

        for i in range(n_step):
            adv = np.sum(delta[i:] * self._actor_discount[:n_step-i])
            advantage[i, action[i]] = adv
            val_error = np.sum(delta[i:] * self._critic_discount[:n_step-i])
            value_target[i, 0] = value[i] + val_error
            # value_target[i, 0] = reward[i] + self.discount * value[i+1]

        self.actor.fit(state, advantage, epochs=1, verbose=0)
        self.critic.fit(state, value_target, epochs=1, verbose=0)

    def train(self, trace):
        state = np.stack(trace['states'][:-1])  # ignore final state
        action = trace['actions']
        reward = trace['rewards']
        n_step = len(state)

        # See Schulman et al. 2016 ICLR paper
        value = np.r_[self.critic.predict(state).flat, 0]  # final state value
        delta = reward + self.discount * value[1:] - value[:-1]  # pg. 4
        advantage = np.zeros((n_step, self.n_action))
        value_target = np.zeros((n_step, 1))

        for i in range(n_step):
            adv = np.sum(delta[i:] * self._actor_discount[:n_step-i])
            advantage[i, action[i]] = adv
            val_error = np.sum(delta[i:] * self._critic_discount[:n_step-i])
            value_target[i, 0] = value[i] + val_error
            # value_target[i, 0] = reward[i] + self.discount * value[i+1]

        self.actor.fit(state, advantage, epochs=1, verbose=0)
        self.critic.fit(state, value_target, epochs=1, verbose=0)


class FixedPlanPolicy(Policy):
    """A policy that blindly executes a fixed sequence of actions."""
    Node = namedtuple('Node', ('state', 'path', 'reward', 'done'))
    def __init__(self, plan, **kwargs):
        super().__init__(**kwargs)
        self._plan = plan

    def start_episode(self, state):
        super().start_episode(state)
        self.plan = iter(self._plan)
        # self.model = Model(self.env)

    def act(self, state):
        return next(self.plan)


class ValSearchPolicy(Policy):
    """Searches for the maximum reward path using a model."""
    def __init__(self, V, replan=False, epsilon=0, noise=1, anneal=1, **kwargs):
        super().__init__(**kwargs)
        self.V = V
        self.replan = replan
        self.epsilon = epsilon
        self.noise = noise
        self.anneal = anneal
        self.history = None
        self.model = None
        self.plan = None

    def start_episode(self, state):
        self.history = Counter()
        self.model = Model(self.env)
        self.plan = iter(())  # start with no plan

    def finish_episode(self, trace):
        self.ep_trace['berries'] = self.env._observe()[-1]

    def act(self, state):
        # return self.env.action_space.sample()
        self.history[state] += 1
        try:
            if self.replan:
                raise StopIteration()
            else:
                return next(self.plan)
        except StopIteration:
            self.plan = iter(self.make_plan(state))
            return next(self.plan)

    def make_plan(self, state, expansions=2000):

        Node = namedtuple('Node', ('state', 'path', 'reward', 'done'))
        env = self.env
        V = memoize(self.V.predict)
        self.node_history = []

        def eval_node(node, noisy=False):
            if not node.path:
                return np.inf  # the empty plan has infinite cost
            obs = env._observe(node.state)
            noise = np.random.rand() * (self.noise * self.anneal ** self.i_episode) if noisy else 0
            value = 0 if node.done else V(obs)[0]
            boredom = - 0.1 * self.history[obs]
            score = node.reward + value + noise + boredom
            return - score

        start = Node(env._state, [], 0, False)
        frontier = PriorityQueue(key=eval_node)
        frontier.push(start)
        reward_to_state = defaultdict(lambda: -np.inf)
        reward_to_state[start.state] = 0
        best_finished = start

        def expand(node):
            nonlocal best_finished
            best_finished = min((best_finished, node), key=eval_node)
            s0, p0, r0, _ = node
            for a, s1, r, done in self.model.options(s0):
                node1 = Node(s1, p0 + [a], r0 + r, done)
                if node1.reward <= reward_to_state[s1]:
                    continue  # cannot be better than an existing node
                self.node_history.append(
                    {'path': node1.path,
                     'r': node1.reward,
                     'b': self.env._observe(node1.state)[-1],
                     'v': -eval_node(node1)})
                reward_to_state[s1] = node1.reward
                if done:
                    best_finished = min((best_finished, node1), key=eval_node)
                else:
                    frontier.push(node1)
                    
        for i in range(expansions):
            if frontier:
                expand(frontier.pop())
            else:
                break



        if frontier:
            plan = min(best_finished, frontier.pop(), key=eval_node)
        else:
            plan = best_finished
        # choices = concat([completed, map(get(1), take(100, frontier))])
        # plan = min(choices, key=eval_node(noisy=True))
        self.log(
            i,
            len(plan.path), 
            -round(eval_node(plan, noisy=False), 2),
            plan.done,
        )
        # self._trace['paths'].append(plan.path)
        return plan.path


class Astar(Policy):
    """A* search finds the shortest path to a goal."""
    def __init__(self, heuristic):
        assert 0  # this implementation is incorrect
        super().__init__()
        self.heuristic = heuristic
        self.plan = iter(())

    def start_episode(self, state):
        self.history = Counter()
        self.model = Model(self.env)

    def act(self, state):
        # return self.env.action_space.sample()
        self.history[state] += 1
        try:
            return next(self.plan)
        except StopIteration:
            self.plan = iter(self.make_plan(state))
            return next(self.plan)

    def eval_node(self, node):
        if not node.path:
            return np.inf  # the empty plan has infinite cost
        obs = self.env._observe(node.state)
        value = 0 if node.done else self.heuristic(self.env, obs)
        boredom = - 0.1 * self.history[obs]
        score = node.reward + value + boredom
        return - score
    
    def make_plan(self, state, expansions=5000):
        
        Node = namedtuple('Node', ('state', 'path', 'reward', 'done'))
        eval_node = self.eval_node
        start = Node(self.env._state, [], 0, False)
        frontier = PriorityQueue(key=eval_node)
        frontier.push(start)
        reward_to_state = defaultdict(lambda: -np.inf)
        # import IPython; IPython.embed()
        best_finished = start

        def expand(node):
            # print(node.state, node.reward, self.rts[node.state], V(env._observe(node.state)))
            # time.sleep(0.1)
            nonlocal best_finished
            # best_finished = min((best_finished, node), key=eval_node)
            s0, p0, r0, _ = node
            for a, s1, r, done in self.model.options(s0):
                node1 = Node(s1, p0 + [a], r0 + r, done)
                if node1.reward <= reward_to_state[s1]:
                    # print('abandon')
                    pass
                    continue  # cannot be better than an existing node
                # self.save('node', node)
                reward_to_state[s1] = node1.reward
                if done:
                    best_finished = min((best_finished, node1), key=eval_node)
                else:
                    frontier.push(node1)
                    
        for i in range(expansions):
            self.save('frontier', [n[1].state for n in frontier])
            if frontier:
                expand(frontier.pop())
            else:
                break

        if frontier:
            # plan = min(best_finished, frontier.pop(), key=eval_node)
            plan = frontier.pop()
            raise RuntimeError('No plan found.')
        else:
            plan = best_finished
        # choices = concat([completed, map(get(1), take(100, frontier))])
        # plan = min(choices, key=eval_node(noisy=True))
        # self.log(
        #     i,
        #     len(plan.path), 
        #     -round(eval_node(plan, noisy=False), 2),
        #     plan.done,
        # )
        # self._trace['paths'].append(plan.path)
        self.save('plan', plan)
        return plan.path