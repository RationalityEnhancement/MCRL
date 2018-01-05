from distributions import Normal
from mouselab import MouselabEnv
import json
import skopt
from policies import LiederPolicy

def filename(cost, note=''):
    c = round(float(cost), 5)
    if note:
        note += '_'
    return f'data/policy_{note}{c}.pkl'

def read_bo_result(cost, note=''):
    return skopt.load(filename(cost, note))

def read_bo_policy(cost, note=''):
    result = read_bo_result(cost, note)
    return LiederPolicy(result.specs['info']['theta'])

def make_env(cost=1.25, ground_truth=None):
    reward = Normal(0, 10).to_discrete(6)
    return MouselabEnv.new_symmetric([4,1,2], reward, cost=cost, ground_truth=None)

ENV = make_env()

def parse_state(state):
    return tuple(ENV.reward if x == '__' else float(x)
                 for x in state)
    
def parse_action(action):
    return ENV.term_action if action == '__TERM_ACTION__' else action

def read_state_actions(cost):
    with open(f'exp-data/human_state_actions_{cost:.2f}.json') as f:
        data = json.load(f)


    return {'states': list(map(parse_state, data['states'])), 
            'actions': list(map(parse_action, data['actions']))}

def render_trace(trace, env=ENV):
    """Saves images of a trace."""
    # from IPython.display import clear_output, display
    # from time import sleep
    from toolz import get
    from shutil import rmtree
    rmtree('trace/', ignore_errors=True)
    for i, (s, a, r) in enumerate(zip(*get(['states', 'actions', 'rewards'], trace))):
        # clear_output()
        env._state = s
        dot = env.render()
        # display(dot)
        dot.render(filename=f'{i}', directory='trace', cleanup=True)
        # sleep(1)
    print('Rendered state sequence to trace/')