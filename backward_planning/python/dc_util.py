import itertools as it
from itertools import product
from collections import Counter, defaultdict, deque
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import copy
sns.set_style('white')
sns.set_context('notebook', font_scale=1.3)

from agents import Agent
from evaluation import get_util
from joblib import Parallel, delayed
from dc_util import *
# from model_utils import *
from mouselab import MouselabEnv
from distributions import Categorical, Normal

def make_envs(cost=1.00, n=100, seed=None,variance_structure="constant_high",branching=[4,1,2]):
    if seed is not None:
        np.random.seed(seed)

    depth = len(branching)

    if variance_structure is "constant_high":
        sigmas = np.concatenate( (np.array([0]),20*np.ones(depth)))
    if variance_structure is "increasing":
        sigmas = [0, 2, 4, 20]
    if variance_structure is "decreasing":
        sigmas = [0,20,4,2]
    if variance_structure is "constant_low":
        sigmas = np.concatenate( (np.array([0]),3*np.ones(depth)))

    def reward(depth):
        if depth > 0:
            return Normal(0, sigmas[depth]).to_discrete(6)
        return 0.

    envs = [MouselabEnv.new_symmetric(branching, reward)
            for _ in range(n)]
    for env in envs:
        env.cost=-cost

    return envs

# fancy initialization function, but doesn't seem necessary
# simpler version built into the class
def fancy_build_path(r):
  if env.tree[r] == []:
      return ([[]], [0])
  paths = []
  path_moves = []
  for n in env.tree[r]:
      new_paths, new_path_moves = build_path(n)
      for i in range(len(new_paths)):
          new_paths[i].insert(0,n)
          new_move = int(hasattr(env._state[n],'sample'))
          path_moves.append(new_path_moves[i]+new_move)
          paths.append(new_paths[i])
  return (paths,path_moves)

def option_util(x,sigma):
    return (sigma*scipy.stats.norm.pdf(x/sigma) -
        np.abs(x)*scipy.stats.norm.cdf(-np.abs(x)/sigma))

def get_all_options_old(env):
    paths = env.paths #list of all paths
    avail_moves = [0,]*len(paths) #list of number of unobserved nodes in each path
    path_obs = [] #value of observed nodes in each path
    path_nodes = [] #the unobserved nodes of each path
    path_stds = [] #the std deviation of the unobserved nodes of each path

    options = [] #list of all options
    option_utils = [] #list of the utility of each option

    for i in range(len(paths)):
        stds = []
        nodes = []
        obs = 0

        for node in paths[i]:
            if hasattr(env._state[node],'sample'):
                stds.append(env._state[node].var())
                nodes.append(node)
                avail_moves[i] += 1
            else:
                obs += env._state[node]

        path_obs.append(obs)
        path_stds.append(stds)
        path_nodes.append(nodes)

        for j in range(avail_moves[i]):
            options.append((i,j+1))
    max_obs = np.max(path_obs)

    for option in options:
        path, obs = option
        option_utils.append(option_util(path_obs[path]-max_obs,np.sqrt(np.sum(np.sort(path_stds[path])[::-1][:obs]))) + obs*env.cost)

    return options, option_utils, path_nodes, path_stds, path_obs, avail_moves

def pick_option_moves(env):
    options, option_utils, path_nodes, path_stds, path_obs, avail_moves = get_all_options_old(env)

    #c is for chosen
    cpath, cobs = options[np.random.choice(np.arange(len(options))[option_utils == np.max(option_utils)])]
    cpath_stds = np.array(path_stds[cpath])
    cpath_nodes = np.array(path_nodes[cpath])
    b = np.random.random(cpath_nodes.size)

    return cpath_nodes[np.lexsort((b,cpath_stds))[::-1][:cobs]]

def all_option_insts(path_nodes,path_stds,n_obs):
    insts = [[]]
    n_remaining_obs = n_obs

    vals, inverse, count = np.unique(path_stds, return_inverse=True,
                                  return_counts=True)
    rows, cols = np.where(inverse == np.arange(len(vals))[:, np.newaxis])
    _, inverse_rows = np.unique(rows, return_index=True)
    res = np.split(cols, inverse_rows[1:])

    for i in range(len(res)):
        new_insts = []

        n_new_nodes = len(res[-i-1])
        if n_new_nodes < n_remaining_obs:
            n_remaining_obs -= n_new_nodes
        else:
            n_new_nodes = n_remaining_obs
            n_remaining_obs = 0

        for new_nodes in it.permutations(res[-i-1],n_new_nodes):
            for inst in insts:
                new_insts.append(inst + list(np.array(path_nodes)[list(new_nodes)]))
        insts = new_insts
        if n_remaining_obs == 0:
            break

    return insts
