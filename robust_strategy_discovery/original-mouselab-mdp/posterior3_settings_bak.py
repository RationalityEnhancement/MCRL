"""
Back-up old settings for posterior3.
"""

from posterior3 import L2M1_PERM, L2H1_PERM, M2H1_PERM, M2L1_PERM, H2M1_PERM, \
    H2L1_PERM, THREE_SAME
from itertools import product

# Original
OUTCOMES = {
    'L': [-2.0, -1.0, 1.0, 2.0],
    'M': [-8.0, -4.0, 4.0, 8.0],
    'H': [-32.0, -16.0, 16.0, 32.0]
}

# ===== Old environments (512) =====
LEVEL_CONFUSION_PROBABILITIES = (
    (('L', 'M'), (0.2, 0.8)),
    (('L', 'M'), (0.3, 0.7)),
    (('M', 'H'), (0.2, 0.8))
)

LEVEL_PREFERENCES = (
    {
        'L': (('L', 'M'), (0.2, 0.8)),
        'M': (('L', 'M'), (0.1, 0.9)),
    },
    {
        'L': (('L', 'M'), (0.3, 0.7)),
        'M': (('L', 'M'), (0.2, 0.8)),
    },
    {
        'M': (('M', 'H'), (0.2, 0.8)),
        'H': (('M', 'H'), (0.1, 0.9))
    },
)

TWO_OBSERVATION_CONFUSION = {
    'L': {
        'L': 0.,
        'M': 0.2,
        'H': 0.1
    },
    'M': {
        'L': 0.8,
        'M': 0.,
        'H': 0.2,
    },
    'H': {
        'L': 0.9,
        'M': 0.8,
        'H': 0.
    }
}

OUTCOMES = {
    'L': [-4.0, -2.0, 2.0, 4.0],
    'M': [-16.0, -8.0, 8.0, 16.0],
    'H': [-48.0, -24.0, 24.0, 48.0]
}

THETAS = list(
    product(L2M1_PERM + M2L1_PERM + [THREE_SAME['M']] + [THREE_SAME['L']],
            L2M1_PERM + M2L1_PERM + [THREE_SAME['M']] + [THREE_SAME['L']],
            H2M1_PERM + M2H1_PERM + [THREE_SAME['H']] + [THREE_SAME['M']])
)
