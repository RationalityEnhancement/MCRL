#!/usr/bin/env python3

import json
from toolz import concat
import numpy as np
import pandas as pd

N_GAMBLE, N_OUTCOME = (7, 4)
directory = 'data/03242018'

def parse_trials(row):
    data = json.loads(row['Answer.data'])
    decisions = map(int, concat(data['decisions']))
    for i, trial in enumerate(concat(data['decision_problems'])):
        X = np.array(trial['reveal_order']).T
        assert X.shape == (N_GAMBLE, N_OUTCOME)
        idx = np.nonzero(X)
        order = np.argsort(X[idx])
        gambles, outcomes = np.transpose(idx)[order].T
        clicks = gambles * N_OUTCOME + outcomes
        yield {
            'trial_index': i,
            'workerid': row.workerid,
            'ground_truth': np.array(trial['payoffs']).astype(float).T.ravel().tolist(),
            'clicks': list(clicks),
            'decision': next(decisions),
            'outcome_probs': trial['probabilities'],
            'reward_mu': trial['mu'][0],
            'reward_sigma': trial['sigma'][0]
        }

df = pd.read_csv(f'{directory}/experiment_results.csv', sep='\t')
trial_df = pd.DataFrame(list(concat(parse_trials(row) for i, row in df.iterrows())))
trial_df.to_csv(f'{directory}/trials.csv')
print(f'Wrote {directory}/trials.csv')