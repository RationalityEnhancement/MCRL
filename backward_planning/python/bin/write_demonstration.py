#!/usr/bin/env python3
from model_utils import *

def optimal_trace(cost):
    agent = Agent()
    agent.register(read_bo_policy(cost))
    envs = make_envs(cost, n=16, ground_truth=1)
    for env in envs:
        agent.register(env)
        trace = agent.run_episode()

        clicks = [{'is_click': 1, 'state': a} for a in trace['actions'][:-1]]
        moves = [{'is_click': 0, 'state': m} for m in env.best_path()]
        yield clicks + moves



demos = {cost: list(optimal_trace(cost))
         for cost in [0.1, 1.25, 4.0]}
with open(f'../experiments/exp1/static/json/demonstrations.json', 'w+') as f:
    json.dump(demos, f)