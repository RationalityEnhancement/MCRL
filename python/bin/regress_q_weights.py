#!/usr/bin/env python3



import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, load, dump
from tqdm import tqdm
from toolz import partition_all, concat

from utils import *
from model_utils import *

N_JOBS = 22
COSTS = [0.1, 1.25, 4.0]


def decode_state(state):
    return tuple(ENV.reward if x == '_' else float(x)
                 for x in state.split())

def get_some_features(cost, sa):
    env = make_env(cost)
    return [env.action_features(a, decode_state(s))
            for s, a in sa]

def get_features(cost, qdf):
    
    unique_sa = list(set(zip(qdf.state, qdf.action)))
    chunks = partition_all(1000, unique_sa)
    jobs = tqdm([delayed(get_some_features)(cost, chunk) for chunk in chunks],
                desc=f'features {cost}')
    return  dict(zip(unique_sa, 
                     concat(Parallel(N_JOBS)(jobs))))

def regress(cost):
    qdf = pd.read_pickle(f'data/q_samples_{cost}.pkl')
    features = get_features(cost, qdf)
    X = np.stack([features[s, a] for s, a in zip(qdf.state, qdf.action)])
    y = np.array(qdf.q)
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    yhat = X @ beta
    
    qmean = qdf.groupby(['state', 'action']).q.mean()
    sa_counts = qdf.groupby(['state', 'action']).q.apply(len)
    qdf = qdf.set_index(['state', 'action'])
    qdf['q_mean'] = qmean[qdf.index]
    qdf['n_visit'] = sa_counts[qdf.index]
    qdf = qdf.reset_index()
    
    qdf['qhat'] = yhat
    qdf['err'] = yhat - qdf.q_mean
    print('MAE:', np.mean(np.abs(qdf.err)))
    return beta, qdf

def main():
    results = {c: regress(c) for c in COSTS}

    with open('data/q_weights.json', 'w+') as f:
        x = {f'{c:.2f}': beta.tolist() for c, (beta, qdf) in results.items()}
        json.dump(x, f)

    print('cost  N     MAE    SEM    beta')
    for c, (beta, qdf) in results.items():
        qdf = qdf.query('n_visit > 199 and action < 17')
        g = qdf.groupby(['state', 'action'])
        sem = (g.q.std() / np.sqrt(g.n_visit.max())).mean()
        mae = np.abs(qdf.err).mean()
        print(f'{c:4} {len(g):4} {mae:.3f}  {sem:.3f}  {beta}')

if __name__ == '__main__':
    main()