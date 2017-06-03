#!/usr/bin/env python3
import json
import sys
sys.path.append('lib')
from utils import dict_product
from datetime import datetime

def standard_to_pseudo(trials):
    def parse_trial(trial):    
        def state(s):
            return {
                'reward': trial['stateRewards'][s],
                'path': trial['path'][s],
                'actions': {k: [int(v[1]), v[0]]
                            for k, v in trial['graph'][s].items()},
            }
        
        return {
            'trialID': trial['trial_i'],
            'initialState': trial['initial'],
            'graph': {s: state(s) for s in trial['graph']}
    }
    return list(map(parse_trial, trials))



def mat_to_standard():

    def parse_trial(t, i):
        graph = {}
        layout = {}
        stateRewards = {}
        path = {}
        # path = {}
        for s in t['states'][0]:
            name = s['nr']

            actions = {}
            sa = s['actions']
            if not isinstance(sa, list):
                sa = [sa]
            for a in sa:
                d = a['direction']
                try:
                    actions[d] = [round(a['reward'], 2), a['state']]
                except:
                    import ipdb, time; ipdb.set_trace(); time.sleep(0.5)
            graph[name] = actions

            x, y = s['location']
            y *= -1
            x += 2
            y += 2
            layout[name] = [x, y]

            stateRewards[name] = round(s['reward'], 2)
            path[name] = s['path']

        return {
            'trial_i': i,
            'initial': 1,
            'layout': layout,
            'graph': graph,
            'stateRewards': stateRewards,
            'stateDisplay': 'click',
            'stateLabels': 'reward',
            'edgeDisplay': 'never',
            'path': path,
        }


    # exp = int(sys.argv[1])
    for cost in ('low', 'med', 'high'):#[:-1]:
        infile = 'data/stimuli/{}_cost.json'.format(cost)
        outfile = 'exp1/static/json/{}_cost.json'.format(cost)
        pseudofile = 'exp1/static/json/pseudo_{}_cost.json'.format(cost)

        with open(infile) as f:
            print('reading', infile)
            data = json.load(f)
        
        trials = [parse_trial(t, i) for i, t in enumerate(data)]

        conditions = list(dict_product({
            'info_cost': [0.01, 1.60, 2.80],
            'PR_type': 2
        }))
        cond = {
            'params': {
                'creation_date': str(datetime.now()),
            },
            'blocks': {'standard': trials},
            'conditions': conditions
        }
        with open(outfile, 'w+') as f:
            print('writing', outfile)
            json.dump(cond, f)

        pseudo = standard_to_pseudo(trials)
        with open(pseudofile, 'w+') as f:
            print('writing', pseudofile)
            json.dump(pseudo, f)


if __name__ == '__main__':
    mat_to_standard()
