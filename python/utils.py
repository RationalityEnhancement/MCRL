from IPython.display import clear_output
import itertools as it

def join(*args, sep=' '):
    return sep.join(map(str, args))

def log_return(func):
    def wrapped(*args, **kwargs):
        r = func(*args, **kwargs)
        print('{} returns {}'.format(func.__name__, r))
        return r
    return wrapped

def logged(condition=lambda r: True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if condition(result):
                print(func.__name__, args, kwargs, '->', result)
            return result
        return wrapper
    return decorator



def clear_screen():
    print(chr(27) + "[2J")
    clear_output()

import time
def show_path(env, trace, render='human'):
    env.reset()
    env.render(mode=render)
    for a in trace['actions']:
        env.step(a)
        input('>')
        env.render(mode=render)


import heapq
class PriorityQueue(list):
    def __init__(self, key, max_first=True):
        self.key = key
        self.inv = -1 if max_first else 1

    def pop(self):
        return heapq.heappop(self)[1]
        
    def push(self, item):
        heapq.heappush(self, (self.inv * self.key(item), item))

def softmax(x):
    ex = np.exp(x)
    return ex / ex.sum()


    # def encode(x):
    #     s = 0
    #     for f, n in zip(x, fs):
    #         s *= n
    #         s += f
    #     return s
            
    # def decode(s):
    #     x = []
    #     for n in reversed(fs):
    #         x.append(s % n)
    #         s //= n
    #     return tuple(reversed(x))
        # super().__init__()

def dict_product(d):
    """All possible combinations of values in lists in `d`"""
    for k, v in d.items():
        if not isinstance(v, list):
            d[k] = [v]

    for v in list(it.product(*d.values())):
        yield dict(zip(d.keys(), v))