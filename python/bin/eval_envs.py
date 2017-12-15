#!/usr/bin/env python3

"""Creates eval_envs.pkl, a DataFrame with policy performance on many environments."""

from evaluation import *
from model_utils import *
from utils import *

import pandas as pd
from policies import FixedPlanPolicy
from joblib import delayed, Parallel
from tqdm import tqdm
from toolz import partition_all

COSTS = [0.1, 1.25, 4.]
term_action = 17

def eval_envs(cost, envs):
    policies = {
        'BO': read_bo_policy(cost),
        'full-observation': FixedPlanPolicy(range(1, term_action+1)),
        'none': FixedPlanPolicy([term_action]),
    }
    def loop():
        for name, pol in policies.items():
            for _ in range(30):
                df = evaluate(pol, envs)
                df['env'] = envs
                df['agent'] = name
                df['cost'] = cost
                yield df
    return pd.concat(loop())



envs_per_cost = 5000
chunks_per_job = 5
total_envs = envs_per_cost * len(COSTS)
n_job = 22
envs_per_chunk = total_envs // (n_job * chunks_per_job)


jobs = [delayed(eval_envs)(cost, envs) 
        for cost in COSTS
        for envs in partition_all(envs_per_chunk, 
                                  make_envs(cost, envs_per_cost, 
                                            sample_term_reward=True, ground_truth=10))]
result = Parallel(n_job)(tqdm(jobs))
pd.concat(result).to_pickle('data/eval_envs.pkl')