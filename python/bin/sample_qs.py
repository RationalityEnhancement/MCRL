#!/usr/bin/env python3
import pandas as pd
from joblib import Parallel, delayed, load, dump
from tqdm import tqdm

from agents import Agent
from policies import LiederPolicy
import skopt
import os

from evaluation import *
from model_utils import *
from utils import *


# We train the LC policy on environments with random reward structures.
# However, to ensure that the policy is near-optimal on belief states
# that participants find themselves in, the initial state is drawn from
# the empirical belief-state distribution of human participants in the
# no-feedback condition.


N_JOBS = 11

COSTS = [0.1, 1.25, 4.0]
# POLICIES = load('data/cross_val_policies.pkl')

def encode_state(state):
    return ' '.join('_' if hasattr(x, 'sample') else str(int(x))
                    for x in state)

def run_rollouts(cost, sa, N=300):
    agent = Agent()
    agent.register(read_bo_policy(cost, note='human_states'))
    states = []
    actions = []
    qs = []
    env = make_env(cost)
    agent.register(env)
    for state, action in sa:
        if action == env.term_action:
            r = env.expected_term_reward(state)
            states.extend([state] * N)
            actions.extend([action] * N)
            qs.extend([r] * N)
        else:
            for _ in range(N):
                env._state = state
                s1, r, *_ = env.step(action)
                env.init = s1
                trace = agent.run_episode()
                states.extend([state] + trace['states'][:-1])
                actions.extend([action] + trace['actions'])
                qs.extend(cum_returns([r] + trace['rewards']))
    return list(map(encode_state, states)), actions, qs

def get_q_samples(cost):
    sa = read_state_actions(cost)
    all_sa = list(zip(sa['states'], sa['actions']))
    chunk_size = len(all_sa) // (N_JOBS * 5)
    jobs = tqdm([delayed(run_rollouts)(cost, chunk)
                 for chunk in partition_all(chunk_size, all_sa)], 
                desc=str(cost))
    results = Parallel(N_JOBS)(jobs)
    print('rollouts completed')
    states, actions, qs = [], [], []
    for s, a, q in results:
        states.extend(s)
        actions.extend(a)
        qs.extend(q)
    return pd.DataFrame({'state': states, 'action': actions, 'q': qs, 'cost': cost})


print('Estimating q values by monte carlo.')
for c in COSTS:
    if os.path.isfile(f'data/q_samples_{c}.pkl'):
        print(f'Skipping {c}')
        continue
    get_q_samples(c).to_pickle(f'data/q_samples_{c}.pkl')
