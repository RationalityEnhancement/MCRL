import gym
from bandits_dqn import MetaBanditEnv
from baselines import deepq
import numpy as np
import sys


def test_bandits(n_arm,cost):
    env = MetaBanditEnv(n_arm, 25, cost)
    filename = "bandit_"+str(n_arm)+"_"+str(cost)+"_model.pkl"
    act = deepq.load(filename)
    tot_rew = 0
    for _ in range(2000):
        obs, done = env.reset(), False
        temp_rew = 0
        while not done:
            obs, rew, done, _ = env.step(act(obs[None])[0])
            temp_rew += rew
        tot_rew += temp_rew
    print("Episode reward", tot_rew/2000)


if __name__ == '__main__':
    n_arm = int(sys.argv[1])
    cost = float(sys.argv[2])
    print(str(n_arm) + " arms, cost: " + str(cost))
    test_bandits(n_arm,cost)