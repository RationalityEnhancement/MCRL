import os
import numpy as np
import sys 
if __name__ == '__main__':
    t_or_e = sys.argv[1]
    for n_arm in range(2,6):
        for cost in np.logspace(-4, -1, 7):
            command = "python "+t_or_e+"_bandits_dqn.py "+str(n_arm) + " " + str(cost)
            print(command)
            os.system(command)
