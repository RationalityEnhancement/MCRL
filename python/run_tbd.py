import os
import numpy as np
import sys 
from joblib import Parallel, delayed

def run_command(commands,cost,n_arm):
    for stage in commands:
        command = "python "+stage+"_bandits_dqn.py "+str(n_arm) + " " + str(cost)
        print(command)
        os.system(command)

if __name__ == '__main__':
    commands = [sys.argv[1]]
    if commands[0] == 'combined':
        commands = ['train','enjoy']
    Parallel(n_jobs=-2)(delayed(run_command)(commands,cost,n_arm)
                       for n_arm in range(2,3) for cost in np.logspace(-4, -1, 7))

