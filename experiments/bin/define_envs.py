#!/usr/bin/env python3
import numpy as np
import itertools as it
from scipy.io import savemat
import os
import json
from collections import defaultdict



# ---------- Constructing environments ---------- #
DIRECTIONS = ('up', 'right', 'down', 'left')
ACTIONS = dict(zip(DIRECTIONS, it.count()))


BRANCH_DIRS = {
    2: {'up': ('right', 'left'),
        'right': ('up', 'down'),
        'down': ('right', 'left'),
        'left': ('up', 'down'),
        'all': ('right', 'left')},
    3: {'up': ('up', 'right', 'left'),
        'right': ('up', 'right', 'down'),
        'down': ('right', 'down', 'left'),
        'left': ('up', 'down', 'left'),
        'all': DIRECTIONS}
}

def move_xy(x, y, direction, dist=1):
    return {
        'right': (x+dist, y),
        'left': (x-dist, y),
        'down': (x, y+dist),
        'up': (x, y-dist),
    }.get(direction)


def dist(branch, depth):
    """Distance between nodes at a given depth of a tree with given branching factor."""
    if branch == 3:
        return 2 ** (depth - 1)
    else:
        return 2 ** (depth/2 - 0.5)
    
class Layouts:
    def cross(depth):
        graph = {}
        layout = {}
        names = it.count()

        def direct(prev):
            if prev == 'all':
                yield from DIRECTIONS
            else:
                yield prev
        
        def node(d, x, y, prev_dir):
            r = 0  # reward is 0 for now
            name = str(next(names))
            layout[name] = (x, y)
            graph[name] = {}
            if d > 0:
                for direction in direct(prev_dir):
                    x1, y1 = move_xy(x, y, direction, 1)
                    graph[name][direction] = (r, node(d-1, x1, y1, direction))
                                            
            return name
        
        node(depth, 0, 0, 'all')
        return graph, layout


    def tree(branch, depth, first='up', **kwargs):
        graph = {}
        layout = {}
        names = it.count()

        def node(d, x, y, prev_dir):
            r = 0  # reward is 0 for now
            name = str(next(names))
            layout[name] = (x, y)
            graph[name] = {}
            if d > 0:
                for direction in BRANCH_DIRS[branch][prev_dir]:
                    x1, y1 = move_xy(x, y, direction, dist(branch, d))
                    graph[name][direction] = (r, node(d-1, x1, y1, direction))
                                            
            return name

        node(depth, 0, 0, first)
        return graph, layout


def build(kind, **kwargs):
    return getattr(Layouts, kind)(**kwargs)


# ---------- Information about environments ---------- #

def n_step(graph):
    """Number of steps it takes to reach every state."""
    result = {}
    def search(s, n):
        result[s] = n
        for _, s1 in graph[s].values():
            search(s1, n+1)
    search('0', 0)
    return result

def get_paths(graph):
    """The sequences of actions and states that lead to each state."""

    state_paths = {}
    action_paths = {}
    def search(s, spath, apath):
        state_paths[s] = spath
        action_paths[s] = apath
        for a, (_, s1) in graph[s].items():
            search(s1, spath + [s1], apath + [a])
    search('0', ['0'], [])
    return {'state_paths': state_paths, 'action_paths': action_paths}


def transition_matrix(graph):
    """X[s0, s1] is 1 if there is an edge from s0 to s1, else 0."""
    X = np.zeros((len(graph), len(graph)))
    for s0, actions in graph.items():
        for _, s1 in actions.values():
            X[int(s0), int(s1)] = 1
    return X

def terminal(graph):
    x = np.zeros(len(graph))
    for s, actions in graph.items():
        if not actions:
            x[int(s)] = 1
    return x

def available_actions(graph):
    X = np.zeros((len(graph), len(ACTIONS)))
    for s0, actions in graph.items():
        for a in actions:
            X[int(s0), ACTIONS[a]] = 1
    return X


def trees():
    paths = defaultdict(dict)
    for branch in (2, 3):
        for depth in range(1, 6):
            graph, layout = build('tree', branch, depth)
            name = 'b{}d{}'.format(branch, depth)

            mat_dict = {
                'transition': transition_matrix(graph),
                'initial': 0,
                'terminal': terminal(graph),
                'actions': available_actions(graph),
                'branch': branch,
                'depth': depth,
            }
            savemat('env_data/{}.mat'.format(name), mdict=mat_dict)
            paths[branch][depth] = get_paths(graph)
    
    os.makedirs('env_data', exist_ok=True)
    with open('experiment/static/json/paths.json', 'w+') as f:
        json.dump(paths, f)

def main():
    from pprint import pprint
    pprint(build_cross(2))

if __name__ == '__main__':
    main()
