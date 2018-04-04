#!/usr/bin/env python3

import itertools as it

DIRECTIONS = ('up', 'right', 'down', 'left')
ACTIONS = dict(zip(DIRECTIONS, it.count()))

BRANCH_DIRS = {
    1: {'up': ['up'],
        'right': ['right'],
        'down': ['down'],
        'left': ['left'],},
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
    return 1
    if branch == 3:
        return 2 ** (depth - 1)
    else:
        return 2 ** (depth/2 - 0.5)  


class Layouts:

    # def cross(depth, mu, sigma, n=4, **kwargs):
    #     graph = {}
    #     layout = {}
    #     names = it.count()

    #     def direct(prev):
    #         if prev == 'all':
    #             yield from  ('right', 'down', 'left')
    #         else:
    #             yield prev
        
    #     def node(d, x, y, prev_dir):
    #         r = 0  # reward is 0 for now
    #         name = str(next(names))
    #         layout[name] = (x, y)
    #         graph[name] = {}
    #         if d > 0:
    #             for direction in direct(prev_dir):
    #                 x1, y1 = move_xy(x, y, direction, 1)
    #                 graph[name][direction] = (r, node(d-1, x1, y1, direction))                     
    #         return name

    #     def r_dist():
    #         from scipy.stats import norm
    #         d = norm(mu, sigma)
    #         vals = np.array([-3, -1, 1, 3]) * sigma + mu
    #         probs = [.1, .4, .4, .1]
    #         # vals = np.linspace(-1.5*sigma, 1.5*sigma, n)
    #         # delta = vals[1] - vals[0]
    #         # bins = np.array((-np.inf, *(vals[1:] - delta/2), np.inf))
    #         # probs = np.diff(d.cdf(bins))
    #         return Categorical(vals, probs)

    #     node(depth, 0, 0, 'all')
    #     rewards = r_dist().sample(len(graph)).tolist()
    #     return graph, layout, rewards

    def tree(branch, depth, first='up', **kwargs):
        graph = {}
        layout = {}
        names = map(str, it.count())

        def node(d, x, y, prev_dir):
            name = next(names)
            layout[name] = (x, y)
            graph[name] = {}
            if d > 0:
                for direction in BRANCH_DIRS[branch][prev_dir]:
                    x1, y1 = move_xy(x, y, direction, dist(branch, d))
                    s1 = node(d-1, x1, y1, direction)
                    graph[name][direction] = (0, s1)
                                            
            return name


        node(depth, 0, 0, first)

        return {
            'layout': layout,
            'initial': '0',
            'graph': graph,
        }

    def tree2(branching, first='up', **kwargs):
        graph = {}
        layout = {}
        names = it.count()

        def node(d, x, y, prev_dir):
            name = str(next(names))
            layout[name] = (x, y)
            graph[name] = {}
            try:
                branch = branching[d]
            except IndexError:
                pass
            else:
                for direction in BRANCH_DIRS[branch][prev_dir]:
                    x1, y1 = move_xy(x, y, direction, dist(branch, d))
                    s1 = node(d+1, x1, y1, direction)
                    if s1 is not None:
                        graph[name][direction] = (0, s1)
                                            
            return name

        node(0, 0, 0, first)
        return {
            'layout': layout,
            'initial': '0',
            'graph': graph,
        }




def main():
    import json
    struct = Layouts.tree2([3,1,2], first='up')
    with open('experiment/static/json/structure/312.json', 'w+') as f:
        json.dump(struct, f)

if __name__ == '__main__':
    main()

