import gym
from bandits_dqn import MetaBanditEnv
from baselines import deepq
import os.path
import sys
import time


def main(n_arm,cost):
    # env = gym.make("MountainCar-v0")
    env = MetaBanditEnv(n_arm, 25, cost)
    # Enabling layer_norm here is import for parameter space noise!
    model = deepq.models.mlp([64], layer_norm=True)
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=2*25*100*1000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        print_freq=1000,
        param_noise=True
    )
    
    filename = "bandit_"+str(n_arm)+"_"+str(cost)+"_model.pkl"
    print("Saving model to " + filename)
    act.save("data/bandit_dqn/weights/"+filename)

if __name__ == '__main__':
    start = time.time()
    n_arm = int(sys.argv[1])
    cost = float(sys.argv[2])
    print(str(n_arm) + " arms, cost: " + str(cost))

    filename = "bandit_"+str(n_arm)+"_"+str(cost)+"_model.pkl"
    if os.path.isfile("data/bandit_dqn/weights/"+filename):
        print('file already exists')
        exit()

    main(n_arm,cost)
    print("--- %s seconds ---" % (time.time() - start))

