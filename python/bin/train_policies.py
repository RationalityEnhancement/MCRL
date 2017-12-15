#!/usr/bin/env python3
import skopt
import joblib

from utils import *
from model_utils import *
from evaluation import *

import warnings
warnings.filterwarnings("ignore", 
                        message="The objective has been evaluated at this point before.")

COSTS = [0.1, 1.25, 4.0]
N_JOBS = 22
N_TRAIN = 880
N_CROSS_VAL = 3520
N_CALLS = 80
NORMALIZE = True

def write_bo_policy(cost, cross_val=True):
    empirical_states = read_state_actions(cost)['states']
    train_envs = make_envs(cost, initial_states=empirical_states, n=N_TRAIN)
    pol, result = bo_policy(train_envs, max_cost=len(train_envs[0].tree),
                            normalize_voi=True, n_random_starts=10,
                            n_calls=N_CALLS, n_jobs=N_JOBS, return_result=True,)
    result.specs['args'].pop('func')  # can't pickle
    result.specs['info'] = {
        'cost': cost,
        'n_train': N_TRAIN,
        'n_calls': N_CALLS,
        'theta': pol.theta
    }

    if cross_val:
        cross_envs = make_envs(cost, initial_states=empirical_states, n=N_CROSS_VAL)
        n_consider = 5
        idx = result.func_vals.argsort()[:n_consider]
        top_x = np.array(result.x_iters)[idx]
        top_theta = [x2theta(x, True) for x in top_x]
        theta = max(top_theta, key=
                    lambda th: get_util(LiederPolicy(th), cross_envs, parallel=joblib.Parallel(N_JOBS)))
        result.specs['info']['theta'] = theta
    skopt.dump(result, filename(cost, note='human_states'))
    return result




print('Training LC policies')
for c in COSTS:
    write_bo_policy(c)
    print('done', c)
