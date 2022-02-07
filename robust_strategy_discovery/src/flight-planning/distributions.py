import numpy as np
from collections import Counter
import itertools as it
from toolz import reduce
import scipy.stats
from functools import total_ordering, partial, lru_cache

LARGE_CACHE_SIZE = int(2**20)
CACHE_SIZE = int(2**14)
SMALL_CACHE_SIZE = int(2**16)

class Distribution(object):
    """Represents a probability distribution."""
    def __add__(self, other):
        raise NotImplementedError()

    def expectation(self):
        raise NotImplementedError()

    def sample(self, n=None):
        raise NotImplementedError()


class Normal(Distribution):
    """Normal distribution."""
    def __init__(self, mu, sigma):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def __repr__(self):
        return 'Norm({:.2f}, {:.2f})'.format(self.mu, self.sigma)

    def __add__(self, other):
        if hasattr(other, 'mu'):
            return Normal(self.mu + other.mu,
                          (self.sigma ** 2 + other.sigma ** 2) ** 0.5)
        # if isinstance(other, PointMass):
        if hasattr(other, 'val'):
            return Normal(self.mu + other.val, self.sigma)
        else:
            return Normal(self.mu + other, self.sigma)
        
    def __radd__(self, other):
        # if isinstance(other, Normal):
        if hasattr(other, 'mu'):
            # print('add norm')
            return Normal(self.mu + other.mu,
                          (self.sigma ** 2 + other.sigma ** 2) ** 0.5)
        # if isinstance(other, PointMass):
        if hasattr(other, 'val'):
            # print('add pointmass')
            return Normal(self.mu + other.val, self.sigma)
        else:
            # print('add number')
            return Normal(self.mu + other, self.sigma)
        
    def __mul__(self, other):
        if not(hasattr(other, 'mu')):
            # print('add number')
            return Normal(self.mu * other, self.sigma * other)
        
    def __rmul__(self, other):
        if not(hasattr(other, 'mu')):
            # print('add number')
            return Normal(self.mu * other, self.sigma * other)

    def to_sampledist(self, n=10000):
        d = SampleDist(self.sample(n))
        d.expectation = lambda *args: self.mu
        return d

    def to_discrete(self, n=10):
        d = scipy.stats.norm(self.mu, self.sigma)
        vals = np.linspace(-2*self.sigma+self.mu, 2*self.sigma+self.mu, n)
        delta = vals[1] - vals[0]
        bins = np.array((-np.inf, *(vals[1:] - delta/2), np.inf))
        probs = np.diff(d.cdf(bins))
        return Categorical(vals, probs)

    def expectation(self):
        return self.mu

    def copy(self):
        return Normal(self.mu, self.sigma)

    def sample(self, n=None):
        if n is not None:
            return self.mu + self.sigma * np.random.randn(n)
        else:
            return self.mu + self.sigma * np.random.randn()

    def sample_nocache(self):
        return self.mu + self.sigma * np.random.randn()

    @classmethod
    def fit(cls, samples):
        return cls(*scipy.stats.norm.fit(samples))
    
class TruncatedNormal(Distribution):
    """Normal distribution."""
    def __init__(self, mu, sigma, lower, upper):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.lower = lower
        self.upper = upper

    def __repr__(self):
        string = 'TNorm({:.2f}, {:.2f},[{:.2f},{:.2f}])'.format(self.mu, self.sigma, self.lower, self.upper)
        # print(string)
        return string
    
    def __add__(self, other):
        if hasattr(other, 'mu') and hasattr(other, 'lower'):
            return TruncatedNormal(self.mu + other.mu,
                          (self.sigma ** 2 + other.sigma ** 2) ** 0.5, self.lower + other.lower, self.upper + other.upper)
        if hasattr(other, 'mu'):
            return Normal(self.mu + other.mu,
                          (self.sigma ** 2 + other.sigma ** 2) ** 0.5)
        # if isinstance(other, PointMass):
        if hasattr(other, 'val'):
            return TruncatedNormal(self.mu + other.val, self.sigma, self.lower + other.val, self.upper + other.val)
        else:
            return TruncatedNormal(self.mu + other, self.sigma, self.lower + other, self.upper + other)
        
    def __radd__(self, other):
        if hasattr(other, 'mu') and hasattr(other, 'lower'):
            return TruncatedNormal(self.mu + other.mu,
                          (self.sigma ** 2 + other.sigma ** 2) ** 0.5, self.lower + other.lower, self.upper + other.upper)
        if hasattr(other, 'mu'):
            return Normal(self.mu + other.mu,
                          (self.sigma ** 2 + other.sigma ** 2) ** 0.5)
        # if isinstance(other, PointMass):
        if hasattr(other, 'val'):
            return TruncatedNormal(self.mu + other.val, self.sigma, self.lower + other.val, self.upper + other.val)
        else:
            return TruncatedNormal(self.mu + other, self.sigma, self.lower + other, self.upper + other)
        
    def __mul__(self, other):
        if not(hasattr(other, 'mu')):
            # print('add number')
            return TruncatedNormal(self.mu * other, self.sigma * other, self.lower * other, self.upper * other)
        
    def __rmul__(self, other):
        if not(hasattr(other, 'mu')):
            # print('add number')
            return TruncatedNormal(self.mu * other, self.sigma * other, self.lower * other, self.upper * other)
        
    def to_sampledist(self, n=10000):
        d = SampleDist(self.sample(n))
        d.expectation = lambda *args: self.mu
        return d

    def expectation(self):
        return self.mu

    def copy(self):
        return TruncatedNormal(self.mu, self.sigma, self.lower, self.upper)

    def sample(self, n=None):
        a , b = (self.lower - self.mu) / self.sigma, (self.upper - self.mu) / self.sigma
        return scipy.stats.truncnorm.rvs(a, b, loc = self.mu, scale = self.sigma, size=n)

class NormalMixture(Distribution):
    """Normal distribution."""
    def __init__(self, mu, sigma, weights):
        super().__init__()
        self.mu = np.array(mu)
        self.sigma = np.array(sigma)
        self.weights = np.array(weights)
        self.n_mix = len(weights)
        self._z = scipy.stats.multinomial(1, weights)
        self._norm = scipy.stats.norm(mu, sigma)

    def __repr__(self):
        return 'NormMix'

    @classmethod
    def mix(cls, dists, weights):
        dists = [d if hasattr(d, 'sample') else Normal(d, 0)
                 for d in dists]
        return cls([d.mu for d in dists], [d.sigma for d in dists], weights)

    def to_sampledist(self, n=10000):
        d = SampleDist(self.sample(n))
        ev = self.mu @ self.weights
        d.expectation = lambda: ev
        return d

    def expectation(self):
        return np.dot(self.mu, self.weights)

    def variance(self):
        return (np.dot(self.weights, (self.sigma ** 2 + self.mu ** 2))
                - self.expectation() ** 2)
        # sigma^2_{g} = sigma^2_R*sum_{o in unobserved(g)} p_o^2

    def copy(self):
        return NormalMixture(self.mu, self.sigma, self.weights)

    def sample(self, n):
        if n is not None:
            z = self._z.rvs(n)
            return (self._norm.rvs((n, self.n_mix)) * z).sum(1)
        else:
            assert False

    def sample_nocache(self):
        return self.mu + self.sigma * np.random.randn()

    @classmethod
    def fit(cls, samples):
        return cls(*scipy.stats.norm.fit(samples))


@total_ordering
class Categorical(Distribution):
    """Categorical distribution."""
    def __init__(self, vals, probs=None):
        super().__init__()
        self.vals = tuple(vals)
        if probs is None:
            self.probs = tuple(1/len(vals) for _ in range(len(vals)))
        else:
            self.probs = tuple(probs)

        # self._hash = id(self)
        self._hash = hash((self.vals, self.probs))

    @lru_cache(None)
    def var(self):
        return sum(v ** 2 * p for v, p in self) - self.expectation() ** 2
    
    @lru_cache(None)
    def std(self):
        return self.var() ** 0.5

    def __lt__(self, other):
        # This is for sorting belief states.
        return True

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return hasattr(other, 'sample')

    def __repr__(self):
        return 'Cat'

    def __str__(self):
        return 'Categorical(vals={}, probs={})'.format(
            np.array(self.vals), np.array(self.probs))

    def __iter__(self):
        return zip(self.vals, self.probs)

    def __len__(self):
        return len(self.probs)

    @lru_cache(maxsize=None)
    def __add__(self, other):
        if hasattr(other, 'probs'):
            # print(f'add({id(self) % 1000}, {id(other) % 1000})')
            return cross((self, other), lambda s, o: s + o)
        if hasattr(other, 'val'):
            return self.apply(lambda v: v + other.val)
        else:
            return self.apply(lambda v: v + other)

    def apply(self, f):
        vals = tuple(f(v) for v in self.vals)
        return Categorical(vals, self.probs)

    @lru_cache(LARGE_CACHE_SIZE)
    def expectation(self):
        return sum(p * v for p, v in zip(self.probs, self.vals))

    def sample(self, n=None):
        vals = np.array(self.vals)  # for use in sample()
        if n is not None:
            i = np.random.choice(len(self.vals), p=self.probs, size=n)
            return vals[i]
        else:
            i = np.random.choice(len(self.vals), p=self.probs)
            return vals[i]


class PointMass(Categorical):
    """A distribution with all mass on one value."""
    def __init__(self, val):
        super().__init__([val], [1])
        self._samples = [val]

    def __hash__(self):
        return hash(self.vals[0])

    def __repr__(self):
        return 'P({})'.format(round(self.vals[0], 2))
        # return 'PointMass({})'.format(round(self.vals[0], 2))

    def __add__(self, other):
        # if isinstance(other, Distribution):
        if hasattr(other, 'sample'):
            return other + self.vals[0]
        else:
            return PointMass(other + self.vals[0])

    def expectation(self):
        return self.vals[0]

    def sample(self, n=None):
        return self.vals[0]


class ScipyDistribution(Distribution):
    """Distribution based on a distribution in scipy.stats,"""
    def sample(self, n=None):
        return self._dist.rvs(n)

    def expectation(self):
        return self._dist.expect()


class Beta(ScipyDistribution):
    """Distribution over [0,1]."""
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self._dist = scipy.stats.beta(alpha, beta)

    def __repr__(self):
        return 'Beta({:.2f}, {:.2f})'.format(self.alpha, self.beta)

    def observe(self, val):
        if val:
            return Beta(self.alpha + 1, self.beta)
        else:
            return Beta(self.alpha, self.beta + 1)


class GenerativeModel(Distribution):
    """Distribution represented by a generative model."""
    def __init__(self, sample, kind=None, args=None):
        super().__init__()
        self._sample = sample
        self.kind = kind
        self.args = tuple(args)

    def __repr__(self):
        if self.kind:
            return '{}{}'.format(self.kind, self.args)

    def __add__(self, other):
        if hasattr(other, 'sample'):
            def sample(n=None):
                return self.sample(n) + other.sample(n)
        else:
            def sample(n=None):
                return self.sample(n) + other
        return GenerativeModel(sample, kind='add', args=(self, other))

    @lru_cache(maxsize=CACHE_SIZE)
    def sample(self, n=None):
        # print('sample', str(self))
        return self._sample(n)

    def expectation(self, n=10000):
        return self.sample(n).mean()


# @lru_cache(maxsize=CACHE_SIZE)
def expectation(val):
    try:
        return val.expectation()
    except AttributeError:
        return val


ZERO = PointMass(0)

# def expectation(val):
#     try:
#         return val.expectation()
#     except AttributeError:
#         return val

def sample(val):
    try:
        return val.sample()
    except AttributeError:
        return val


# def cross(d1, d2, f=None):
#     if f is None:
#         f = lambda *args: args
#     outcomes = Counter()
#     for ((o1, p1), (o2, p2)) in it.product(d1, d2):
#         outcomes[f(o1, o2)] += p1 * p2

#     return Categorical(outcomes.keys(), outcomes.values())

def cross(dists, f=None, verbose=False):
    """
    For a given distribution, finds best possible values and probabilities.
    Returns a categorical distribution of new values.
    """
    # if(verbose):
    print("In Cross, Dist = {}".format(*dists))
    if f is None:
        f = lambda *x: x
    outcomes = Counter() # Fancy dictionary whose key are possible node value and the corresponsing field value is its probability
    for outcome_probs in it.product(*dists):
        o, p = zip(*outcome_probs)
        outcomes[f(*o)] += reduce(lambda x, y: x*y, p)
        if verbose:
            print("Outcomes = {}".format(outcomes))
    if verbose:
        print("Cross Return: {}".format(Categorical(outcomes.keys(), outcomes.values())))
    return Categorical(outcomes.keys(), outcomes.values())


__no_default__ = 25


# @lru_cache(maxsize=None)
def cmax(dists, verbose=False, default=__no_default__):
    """
    Returns the best possible value for a given distribution. Look at cross function for more clarity!
    """
    dists = tuple(dists)
    if len(dists) == 1:
        return dists[0]
    elif len(dists) == 0:
        if default is not __no_default__:
            return default
        else:
            raise ValueError('dmax() arg is an empty sequence')
    else:
        return cross(dists, max,verbose)


# @lru_cache(maxsize=None)
def dmax(dists, default=__no_default__):
    assert 0 
    dists = tuple(dists)
    if len(dists) == 1:
        return dists[0]
    if len(dists) == 0:
        if default is not __no_default__:
            return default
        else:
            raise ValueError('dmax() arg is an empty sequence')

    def sample(n=None):
        return reduce(np.maximum, [d.sample(n) for d in dists])

    return GenerativeModel(sample, kind='dmax', args=dists)

# @lru_cache(CACHE_SIZE)
def smax(dists, default=__no_default__):
    if len(dists) == 0:
        if default is not __no_default__:
            return default
        else:
            raise ValueError('dmax() arg is an empty sequence')
    elif len(dists) == 1:
        return dists[0]
    elif len(dists) == 2:
        a, b = dists[0]._samples, dists[1]._samples
        if a[0] == b[0]:  # the same samples
            b = np.random.permutation(b)
        return SampleDist(np.maximum(a, b))
    else:
        raise NotImplementedError()
        return SampleDist(reduce(np.maximum, [d._samples for d in dists]))


def normal_approximation(dist, samples=10000):
    return Normal(scipy.stats.norm.fit(dist.sample(samples)))


class SampleDist(Distribution):
    """A distribution represented by samples."""
    def __init__(self, samples):
        super().__init__()
        self._samples = samples
        self.len = len(samples) if hasattr(samples, '__len__') else None

    def __repr__(self):
        return 'SD({})'.format(id(self) % 1000)

    def sample(self):
        return np.random.choice(self._samples)

    def expectation(self):
        return np.mean(self._samples)

    @lru_cache(SMALL_CACHE_SIZE)
    def __add__(self, other):
        if hasattr(other, '_samples'):
            return SampleDist(self._samples + other._samples)
        elif hasattr(other, 'sample'):
            return SampleDist(self._samples + other.sample(self.len))
        else:
            return SampleDist(self._samples + other)
        
    def __radd__(self, other):
        if hasattr(other, '_samples'):
            return SampleDist(self._samples + other._samples)
        elif hasattr(other, 'sample'):
            return SampleDist(self._samples + other.sample(self.len))
        else:
            return SampleDist(self._samples + other)
    
    def __mul__(self, other):
        if hasattr(other, '_samples'):
            return SampleDist(self._samples * other._samples)
        elif hasattr(other, 'sample'):
            return SampleDist(self._samples * other.sample(self.len))
        else:
            return SampleDist(self._samples * other)
    def __rmul__(self, other):
        if hasattr(other, '_samples'):
            return SampleDist(self._samples * other._samples)
        elif hasattr(other, 'sample'):
            return SampleDist(self._samples * other.sample(self.len))
        else:
            return SampleDist(self._samples * other)
