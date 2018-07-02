import gym
from bandits_dqn import MetaBanditEnv
from baselines import deepq
import numpy as np
import sys
import pandas as pd

def test_bandits(n_arm,cost):
    env = MetaBanditEnv(n_arm, 25, cost)
    cost_i = np.abs(np.logspace(-4, -1, 7) - cost).argmin()
    filename = "data/bandit_dqn/weights/bandit_"+str(n_arm)+"_"+str(cost)+"_model.pkl"
    act = deepq.load(filename)
    tot_rew = 0
    dfs = []
    for _ in range(2000):
        obs, done = env.reset(), False
        temp_rew = 0
        obs_count = 0
        while not done:
            obs, rew, done, _ = env.step(act(obs[None])[0])
            temp_rew += rew
            obs_count += 1
        df = {'util' : temp_rew, 'observations': obs_count - 1,
              'agent' : 'dqn', 'n_arm' : n_arm, 'max_obs' : 25,
              'cost' : cost}
        tot_rew += temp_rew
        dfs.append(df)
    print(str(n_arm)+ " arm, cost: "+str(cost)+", reward: " + str(tot_rew/2000))
    data = pd.DataFrame(dfs)
    print(data.util.mean())
    store = pd.HDFStore('data/bandit_dqn/results/dqn_results_'+str(n_arm)+"_"+str(cost_i)+'.h5')
    store['data'] = data
    store.close()

if __name__ == '__main__':
    n_arm = int(sys.argv[1])
    cost = float(sys.argv[2])
    print(str(n_arm) + " arms, cost: " + str(cost))
    test_bandits(n_arm,cost)