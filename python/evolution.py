import pygmo as pg
import numpy as np


class EvolutionaryOptimizer(object):
    """Minimizes a function with an evolutionary strategy."""
    def __init__(self, func, bounds, size=30, gen=100, algo=None):
        super().__init__()
        if algo is None:
            algo = pg.cmaes(gen=gen)
        
        self.func = func
        self.bounds = bounds
        self.gen = gen
        self.xs = []
        self.fs = []
      
        class problem:
            def __init__(self):
                super().__init__()
                self._seed = 0
            def fitness(self, x):
                np.random.seed(self._seed)
                return [func(x)]
            def get_bounds(self):
                return bounds
            def set_seed(self, seed):
                self._seed = seed

        self.prob = pg.problem(problem())
        self.algo = pg.algorithm(algo)
        self.pop = pg.population(self.prob, size)
        self.xs.append(self.pop.get_x())
        self.fs.append(self.pop.get_f().squeeze())


    def evolve(self):
        print('evolve')
        self.pop = self.algo.evolve(self.pop)
        self.xs.append(self.pop.get_x())
        self.fs.append(self.pop.get_f().squeeze())


bounds = [
    (0, 100), # is_term
    (0, 10),   # term_reward
    (0, 0),   # voi_myopic
    (0, 0),   # vpi_action
    (0, 0),   # vpi_full
    (0, 10),  # quality_ev
    (0, 10),  # quality_std
    (0, 100), # depth
]