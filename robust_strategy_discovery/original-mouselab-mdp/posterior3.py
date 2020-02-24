from datetime import datetime
from itertools import permutations, product
from pprint import pprint
from random import choice
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

TRAIN = 1
LOAD = 2

# Additional
LEVEL_LENGTH = (
    3, 3, 3
)

# Random remembering
LEVEL_CONFUSION_PROBABILITIES = (
    (('L', None), (1.0, 0.0)),
    (('L', None), (1.0, 0.0)),
    (('M', 'H'), (0.1, 0.9))
)

# Confused remembering
LEVEL_PREFERENCES = (
    {
        'L': (('L', None), (1.0, 0.0)),
    },
    {
        'M': (('M', None), (1.0, 0.0)),
    },
    {
        'M': (('M', 'H'), (0.45, 0.55)),
        'H': (('M', 'H'), (0.10, 0.90))
    }
)

# Preference matrix | P_MTX[e1][e2] = Probability of "remembering" e1 over e2
TWO_OBSERVATION_CONFUSION = {
    'L': {
        'L': 0.,
        'M': 0.2,
        'H': 0.1
    },
    'M': {
        'L': 0.8,
        'M': 0.,
        'H': 0.4,
    },
    'H': {
        'L': 0.9,
        'M': 0.6,
        'H': 0.
    }
}

# Variance types
NODE_TYPES = {
    'L': 0,  # "Value" the low-variance node type
    'M': 1,  # "Value" the middle-variance node type
    'H': 2   # "Value" the high-variance node type
}

# Possible outcomes
OUTCOMES = {
    'L': [-4.0, -2.0, 2.0, 4.0],
    'M': [-8.0, -4.0, 4.0, 8.0],
    'H': [-48.0, -24.0, 24.0, 48.0]
}

# Generate theta space
HML_PERM = [
    ('L', 'M', 'H'),
    ('L', 'H', 'M'),
    ('M', 'L', 'H'),
    ('M', 'H', 'L'),
    ('H', 'L', 'M'),
    ('H', 'M', 'L')
]

# Two-type permutations
H2M1_PERM = [
    ('H', 'H', 'M'),
    ('H', 'M', 'H'),
    ('M', 'H', 'H')
]

H2L1_PERM = [
    ('H', 'H', 'L'),
    ('H', 'L', 'H'),
    ('L', 'H', 'H')
]

M2H1_PERM = [
    ('M', 'M', 'H'),
    ('M', 'H', 'M'),
    ('H', 'M', 'M')
]

M2L1_PERM = [
    ('M', 'M', 'L'),
    ('M', 'L', 'M'),
    ('L', 'M', 'M')
]

L2H1_PERM = [
    ('L', 'L', 'H'),
    ('L', 'H', 'L'),
    ('H', 'L', 'L')
]

L2M1_PERM = [
    ('L', 'L', 'M'),
    ('L', 'M', 'L'),
    ('M', 'L', 'L')
]

THREE_SAME = {
    'H': ('H', 'H', 'H'),
    'M': ('M', 'M', 'M'),
    'L': ('L', 'L', 'L')
}

# New environment (8 environments)
THETAS = list(
    product([THREE_SAME['L']],
            [THREE_SAME['M']],
            [THREE_SAME['M']] + M2H1_PERM + H2M1_PERM + [THREE_SAME['H']])
)

ENUM_THETA = {
    idx: theta
    for idx, theta in enumerate(THETAS)
}

ENUM_THETA_FLAT = dict()
REV_ENUM_THETA = dict()

for key, vector in ENUM_THETA.items():
    theta_string = ''
    for level_vector in vector:
        theta_string += ''.join(level_vector)
    
    ENUM_THETA_FLAT[key] = theta_string
    REV_ENUM_THETA[theta_string] = key


# Functions
def env_to_one_hot(vector):
    """
    Generates element-wise one hot vectors.
    Example: (H, L, M) --> (2, 0, 1) --> ((0, 0, 1), (1, 0, 0), (0, 1, 0))
    """
    flat_vector = []
    for level_vector in vector:
        flat_vector += list(level_vector)
    
    one_hot = []
    for item in flat_vector:
        value = get_value(item)
        one_hot += number_to_one_hot(value, len(NODE_TYPES))
    
    return one_hot


def get_theta(idx=None):
    # - Get no theta index specified
    if idx is None:
        idx = choice(np.arange(len(THETAS)))
    
    # - Get values of theta
    theta = THETAS[idx]
    
    # - Encode theta index to 1-hot vector
    theta_one_hot = number_to_one_hot(idx, len(THETAS))
    
    return theta, theta_one_hot


# TODO: Make process models into one function
def get_confused_theta_hat(theta):
    # Generate biased environment (theta_hat)
    theta_hat = []
    for level_idx, level_nodes in enumerate(theta):
        theta_hat += confused_remembering(level_nodes, level_idx)

    # Encode theta_hat into element-wise 1-hot vector
    theta_hat_one_hot = env_to_one_hot(theta_hat)

    return theta_hat, theta_hat_one_hot


def get_random_theta_hat():
    # Generate biased environment (theta_hat)
    theta_hat = []
    for idx in range(len(LEVEL_LENGTH)):
        node_types, prob = LEVEL_CONFUSION_PROBABILITIES[idx]
        vector_size = LEVEL_LENGTH[idx]
        theta_hat += random_remembering(node_types, prob, vector_size)
    
    # Encode theta_hat into element-wise 1-hot vector
    theta_hat_one_hot = env_to_one_hot(theta_hat)
    
    return theta_hat, theta_hat_one_hot


def get_two_observation_theta_hat(theta):
    # Generate biased environment (theta_hat)
    theta_hat = []
    for level in theta:
        theta_hat += two_observation_remembering(level)

    # Encode theta_hat into element-wise 1-hot vector
    theta_hat_one_hot = env_to_one_hot(theta_hat)

    return theta_hat, theta_hat_one_hot


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H_%M_%S")


def get_value(node_type):
    return NODE_TYPES[node_type]


# TODO: Add theta_hat function as a parameter
def generate_data(n_items=1, verbose=False):
    print(f'Generating {n_items} items of data...')
    
    items = []
    
    theta_hats = []
    
    for _ in tqdm(range(n_items)):
        # Get real environment (theta)
        theta, theta_one_hot = get_theta()
        
        # Get biased environment | # TODO: Change this before run!
        theta_hat, theta_hat_one_hot = get_confused_theta_hat(theta)
        # theta_hat, theta_hat_one_hot = get_random_theta_hat()
        # theta_hat, theta_hat_one_hot = get_two_observation_theta_hat(theta)

        # Append theta_hat vector
        theta_hats += [theta_hat]

        # Print if necessary
        if verbose:
            print('Theta:', theta)
            print('Theta 1-hot:', theta_one_hot, '\n')
            
            print('Theta_hat:', theta_hat)
            print('Theta_hat 1-hot:', theta_hat_one_hot, '\n')
        
        # Append new data item to the list of data items
        items += [np.array([
            np.array(theta_hat_one_hot),
            np.array(theta_one_hot)
        ])]
    
    # Cast items into numpy array
    items = np.array(items)
    
    # Split input and output data
    data = np.stack(items[:, 0])
    labels = np.stack(items[:, 1])
    
    data, labels = shuffle(data, labels)
    
    return data, labels, theta_hats


def generate_path(path, with_timestamp=False):
    if with_timestamp:
        timestamp = get_timestamp()
        path = f'{path}{timestamp}/'
    os.makedirs(path, exist_ok=True)
    
    return path


def number_to_one_hot(num, max_len):
    one_hot = np.zeros(max_len, dtype=np.int)
    one_hot[num] = 1
    return list(one_hot)


def confused_remembering(node_types, level_idx):
    """
    ...
    """
    prob = LEVEL_PREFERENCES[level_idx]
    
    belief = [None for _ in range(len(node_types))]
    
    for idx, node_type in enumerate(node_types):
        possible_node_types, p = prob[node_type]
        belief[idx] = np.random.choice(possible_node_types, p=p)
    
    return tuple(belief)


def one_observation_remembering(node_types):
    """
    Generates an assumption based on only 1 observation (uncovered node).
    """
    observation = np.random.choice(node_types)
    return tuple(observation) * len(node_types)


def random_remembering(node_types, p, vector_size):
    belief = [None for _ in range(vector_size)]
    
    for idx in range(vector_size):
        belief[idx] = np.random.choice(node_types, p=p)
    
    return tuple(belief)


def two_observation_remembering(node_types):
    """
    Generates an assumption based on 2 observations (uncovered nodes).
    """
    idx = np.random.choice(np.arange(len(node_types)), 2, replace=False)
    other_idx = set(np.arange(len(node_types)))
    other_idx.difference_update(set(idx))

    # Get node types
    n1 = node_types[idx[0]]
    n2 = node_types[idx[1]]

    # Compare both values
    v1 = get_value(n1)
    v2 = get_value(n2)

    belief = [None for _ in range(len(node_types))]
    belief[idx[0]] = n1
    belief[idx[1]] = n2

    # Get probability of choosing n1 over n2
    prob = TWO_OBSERVATION_CONFUSION[n1][n2]

    if v1 > v2:
        for i in other_idx:
            belief[i] = np.random.choice([n1, n2], p=[prob, 1 - prob])
    elif v1 < v2:
        for i in other_idx:
            belief[i] = np.random.choice([n1, n2], p=[prob, 1 - prob])
    else:  # Same value appeared twice
        for i in other_idx:
            belief[i] = np.random.choice([n1, n2], p=[prob, 1 - prob])

    return tuple(belief)


# Training procedure
def train_process_posterior_function(data, labels, path,
                                     epochs=10, batch_size=512):
    path += '/model.h5'
    
    input_size = len(data[0])
    output_size = len(labels[0])
    
    def build_model():
        # Original model
        # model = keras.Sequential([
        #     layers.Dense(20, activation='relu', input_shape=[input_size]),
        #     layers.Dense(10, activation='relu'),
        #     layers.Dense(10, activation='relu'),
        #     layers.Dense(output_size, activation='softmax'),
        # ])
        
        # Larger model
        model = keras.Sequential([
            layers.Dense(50, activation='relu', input_shape=[input_size]),
            layers.Dense(70, activation='relu'),
            layers.Dense(100, activation='relu'),
            layers.Dense(100, activation='relu'),
            layers.Dense(60, activation='relu'),
            layers.Dense(output_size, activation='softmax'),
        ])

        return model
    
    model = build_model()
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
    )
    
    model.fit(
        data,  # training data
        labels,  # training targets
        epochs=epochs,
        batch_size=batch_size,
    )
    
    model.save(path)
    
    return model


# Testing procedure
def evaluate(model, input_vector):
    return model.predict(input_vector)[0]
    

if __name__ == '__main__':
    # Constants / Configurations
    MODE = TRAIN
    N_PLOTS = len(THETAS)
    
    SAMPLE_FACTOR = 1250
    NUM_SAMPLES = len(THETAS) * SAMPLE_FACTOR
    NUM_EPOCHS = 20
    
    PATH = f'posterior3/confused_remebemring/' \
           f'{SAMPLE_FACTOR}x_samples_{NUM_EPOCHS}_epochs'

    # Variables
    accuracy = 0
    posteriors = list()

    # Train new model
    if MODE == TRAIN:
        path = generate_path(PATH)
        data, labels, theta_hats = \
            generate_data(n_items=NUM_SAMPLES)
        model = train_process_posterior_function(data, labels, path,
                                                 epochs=NUM_EPOCHS)

    # Load model
    if MODE == LOAD:
        path = f'{PATH}'
        model = keras.models.load_model(f'{path}/model.h5')

    os.makedirs(f'{path}/plots/{N_PLOTS}/01_scale/', exist_ok=True)
    os.makedirs(f'{path}/plots/{N_PLOTS}/opt_scale/', exist_ok=True)

    # Generate posterior distributions for each theta_hat
    if MODE == TRAIN:
        print('Generating and plotting posterior distributions...')
        for i in tqdm(range(len(THETAS))):
            
            # Get biased environment (theta_hat)
            theta_hat = THETAS[i]
            theta_hat_one_hot = env_to_one_hot(theta_hat)

            # Cast theta(_hat) vector into NumPy array
            theta_hat_one_hot = np.array(theta_hat_one_hot).flatten().reshape([1, -1])
            
            posterior_dist = evaluate(model, theta_hat_one_hot)
    
            posteriors += [posterior_dist]
    
            plt.figure(figsize=(24, 10))
            plt.bar(np.arange(len(posterior_dist)), posterior_dist)
            plt.title(f'Theta_hat idx = {i}')
            plt.savefig(f'{path}/plots/{N_PLOTS}/opt_scale/{i}')

            plt.figure(figsize=(24, 10))
            plt.bar(np.arange(len(posterior_dist)), posterior_dist)
            plt.title(f'Theta_hat idx = {i}')
            plt.ylim(-.1, 1.1)
            plt.savefig(f'{path}/plots/{N_PLOTS}/01_scale/{i}')

        # Save posterior distributions to CSV
        pd.DataFrame(posteriors).to_csv(f"{path}/posteriors.csv",
                                        index_label="theta_hat_idx")
        
    # Check accuracy
    print('Calculating accuracy...')
    for i in tqdm(range(NUM_SAMPLES)):
        # Sample real environment (theta)
        theta, theta_one_hot = get_theta()
        
        # Get biased environment (theta_hat)
        theta_hat, theta_hat_one_hot = get_confused_theta_hat(theta)

        # Cast theta(_hat) vector into NumPy array
        theta_one_hot = np.array(theta_one_hot).flatten().reshape([1, -1])
        theta_hat_one_hot = np.array(theta_hat_one_hot).flatten().reshape([1, -1])

        posterior_dist = evaluate(model, theta_hat_one_hot)
        # print(i, '|', np.round(posterior_dist, 3))
        
        output = np.argmax(posterior_dist)
        target = np.argmax(theta_one_hot)
        if output == target:
            accuracy += 1

    print(f'Correct: {accuracy} | Total {NUM_SAMPLES} | '
          f'Accuracy: {accuracy / NUM_SAMPLES * 100:.3f}%')
