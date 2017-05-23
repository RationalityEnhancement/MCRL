from functools import wraps
import json
import os
from collections import namedtuple, defaultdict
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.propagate = False
ch = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s: %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

class _StimulatorMeta(type):
    """MetaClass that supports the magic of the `stimulus` decorator."""
    def __new__(cls, name, bases, local):
        """Wraps the methods marked as stimuli generators."""
        for attr, value in local.items():
            if callable(value) and hasattr(value, '__stimulus'):
                local[attr] = cls.wrap_stim_method(value)
        return type.__new__(cls, name, bases, local)

    def wrap_stim_method(method):
        """Implements the `stimulus` decorator."""
        dec_args = method.__dict__['__stimulus']
        kind = dec_args['name'] or method.__name__.split('stim_')[-1]

        @wraps(method)
        def wrapped(self, *args):
            # Check if stimulus has already been created.
            identify = dec_args['identify'] or self.identify
            identity = identify(args)
            if identity in self._stims[kind]:
                return self._stims[kind][identity]

            # Create the new stimulus.
            obj = method(self, *args)
            stim_id = '{}_{}'.format(kind, len(self._stims[kind]))


            # Save the stimulus as a json file
            rel_file = os.path.join(self.stim_dir, stim_id + '.json')
            file = os.path.join(self.base_dir, rel_file)
            if hasattr(obj, 'to_json'):
                obj.to_json(file)
            else:
                if hasattr(obj, 'to_dict'):
                    d = obj.to_dict()
                elif isinstance(obj, dict):
                    d = dict(obj)
                else:
                    raise ValueError('Stimulus must be JSON serializable.')
                with open(file, 'w+') as f:
                    if isinstance(d, dict):
                        pop = [k for k in d.keys() if k.startswith('__')]
                        for k in pop:
                            d.pop(k)
                    json.dump(d, f)

            stim = self.Stim(obj, rel_file)
            self._stims[kind][identity] = stim
            self.stims[stim_id] = stim
            log.info('Wrote stimulus: %s', file)
            return stim

        return wrapped


def stimulus(name=None, identify=None):
    """A decorator that marks a function as creating a stimulus.

    The object returned by the decorated function will be saved as a json file.
    Thus, it must be either JSON-serializable or an object implementing a
    `to_json()` method. Although the wrapped function should only return
    the stimulus, this decorator modifies it such that when called it will
    return the stimulus as well as the file where that stimuli was saved.
    The function will be called only once for each unique argument
    configuration. Thus, the same stimuli may be used in multiple trials
    (perhaps in different conditions). If this behavior is not desired,
    you must add an additional parameter (perhaps unused by the function)
    that will make unique the argument configuration, and therefore the
    json file.
    """
    def decorator(method):
        method.__stimulus = {'name': name, 'identify': identify}
        return method
    return decorator


class Stimulator(object, metaclass=_StimulatorMeta):
    """Defines conditions and stimuli for an experiment."""
    Stim = namedtuple('Stim', ['obj', 'file'])

    def __init__(self, base_dir='experiment', stim_dir='static/json'):
        self.base_dir = base_dir
        self.stim_dir = stim_dir
        self.stims = {}  # stim_id -> stim.obj
        self._stims = defaultdict(dict)  # kind, identity -> stim_id
        self.conds = []

    def conditions(self):
        yield {}

    def blocks(self, params):
        yield params

    def trials(self, params):
        yield params

    def trial(self, params):
        return params

    def counter_balance(self, cond):
        yield cond
        
    def run(self):
        for cond_i, cond_params in enumerate(self.conditions()):
            cond_params['cond_i'] = cond_i
            cond = {'params': cond_params, 'blocks': {}}
            self.conds.append(cond)
            for block_i, block_params in enumerate(self.blocks(cond_params)):
                block_name = block_params.get('block', 'block_{}'.format(block_i))
                cond['blocks'][block_name] = block = []
                for trial_i, trial_params in enumerate(self.trials({**cond_params, **block_params})):
                    trial_params['trial'] = trial_i
                    trial = self.trial(trial_params)
                    block.append(trial)
                    print('trial_i = {}'.format(trial_i))

            for counter_i, counter_cond in enumerate(self.counter_balance(cond)):
                cond_file = os.path.join(self.base_dir, self.stim_dir, 
                                         'condition_{}_{}.json'.format(cond_i, counter_i))

                counter_cond['params']['counter_i'] = counter_i

                with open(cond_file, 'w+') as cf:
                    json.dump(counter_cond, cf)
                    log.info('Wrote condition: %s', cond_file)
                    log.debug(counter_cond)
            return self

    @staticmethod
    def identify(args):
        return '_'.join(map(repr, args))




class Test(Stimulator):
    
    def conditions(self):
        yield {'c': 1}
        yield {'c': 2}

    def blocks(self, params):
        yield params

    def trials(self, params):
        for c in 'red', 'blue':
            for n in 1, 3:
                yield {'color': c, 'number': n}

    def trial(self, params):
        # stim = stim_test(params['color'], params['number'])
        stim = self.stim1(params['color'], params['number'])
        return stim.file

    @stimulus()
    def stim1(self, color, number):
        return [color, number]


if __name__ == '__main__':
    Test().run()





