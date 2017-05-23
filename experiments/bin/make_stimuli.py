#!/usr/bin/env python3
from datetime import datetime
import sys
import numpy as np

sys.path.append('lib')
from stimulator import Stimulator
from utils import dict_product, Labeler
from define_envs import build

class Stims(Stimulator):
    """Defines conditions and creates stimuli."""
    

    # ---------- Experiment structure ---------- #

    def conditions(self):
        yield {
            'creation_date': str(datetime.now())
        }
    def blocks(self, params):
        yield {
            'block': 'standard',
        }

    def trials(self, params):
        for d in range(2, 4):
            yield {'depth': d,
                   'kind': 'cross'}

    # ---------- Create stimuli ---------- #

    def trial(self, params):
        params.pop('trial')
        graph, layout = build(**params)
        return {
            'graph': graph,
            'layout': rescale(layout),
            # 'stateLabels': dict(zip(graph.keys(), graph.keys())),
            'stateRewards': dict(zip(graph.keys(), map(int, graph.keys()))),
            'stateLabels': 'reward',
            'stateDisplay': 'click',
            'stateClickCost': 0.05,
            'edgeDisplay': 'never',
            'initial': '0'
        }


def rescale(layout):
    names, xy = zip(*layout.items())
    x, y = np.array(list(xy)).T
    x -= x.min()
    y -= y.min()
    return dict(zip(names, zip(x.tolist(), y.tolist())))


if __name__ == '__main__':
    Stims().run()

