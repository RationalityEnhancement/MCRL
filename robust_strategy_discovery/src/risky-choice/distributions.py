import numpy as np
from collections import Counter
import itertools as it
from toolz import reduce
import scipy.stats
from functools import total_ordering, partial, lru_cache
from scipy.stats import norm, truncnorm
import scipy.integrate as integrate

LARGE_CACHE_SIZE = int(2**20)
CACHE_SIZE = int(2**14)
SMALL_CACHE_SIZE = int(2**16)

standard_norm = norm(0, 1)

class Distribution(object):
    """Represents a probability distribution."""
    def __add__(self, other):
        raise NotImplementedError()

    def expectation(self):
        raise NotImplementedError()

    def sample(self, n=None):
        raise NotImplementedError()


class Mixture(Distribution):
    """A distribution which is a mixture of distributions (need those distributions to implement functions like __mul__ themselves) """
    def __init__(self, distributions, probabilities):        
        super().__init__()
        self.distributions = distributions
        self.probabilities = probabilities

        mu = 0
        for i in range(len(distributions)):
            mu+=expectation(distributions[i])*probabilities[i]

        self.mu = mu    

    def __repr__(self):
        return 'Mix({:.2f})'.format(self.mu)    


    def __str__(self):
        return 'Mixture(distributions={}, probs={})'.format(
            np.array(self.distributions), np.array(self.probabilities))


    def expectation(self):
        return self.mu

    def sample(self, n=None):
        if n is None:
            index = np.random.choice(len(self.distributions), p = self.probabilities)
            return sample(self.distributions[index])

        else:
            indices = np.random.choice(len(self.distributions), size = n, p = self.probabilities)
            samples = []
            for index in indices:
                samples.append(sample(self.distributions[index]))

            return np.array(samples) 


    def __mul__(self, other):
        if not(hasattr(other,'sample')):
            if abs(other) < 1e-30:
                return 0.0

            else:
                distributions = []
                for dist in self.distributions:
                    distributions.append(other*dist) 
                return Mixture(distributions, self.probabilities)    

    def __rmul__(self, other):
        if not(hasattr(other,'sample')):
            if abs(other) < 1e-30:
                return 0.0

            else:
                distributions = []
                for dist in self.distributions:
                    distributions.append(other*dist) 
                return Mixture(distributions, self.probabilities)    

    def __add__(self, other):        
        if hasattr(other, 'distributions'):
            distributions = []
            probabilities = []
        
            for ind1 in range(len(self.distributions)):
                for ind2 in range(len(other.distributions)):
                    distributions.append(self.distributions[ind1] + other.distributions[ind2])
                    probabilities.append(self.probabilities[ind1]*other.probabilities[ind2])
            return Mixture(distributions, probabilities)

        
        else:
            distributions = []
            for dist in self.distributions:
                distributions.append(dist + other)

            return Mixture(distributions, self.probabilities)
                

    def __radd__(self, other):
        distributions = []
        for dist in self.distributions:
            distributions.append(dist + other)

        return Mixture(distributions, self.probabilities)
    

    def cdf(self, k):
        cdfk = 0.0
        for index in range(len(self.distributions)):
            dist = self.distributions[index]
            cdfk += self.probabilities[index]*dist.cdf(k)

        return cdfk

    def integrate_xfx(self, low = -np.inf, high = np.inf):
        xfx = 0.0
        for index in range(len(self.distributions)):
            dist = self.distributions[index]
            xfx += self.probabilities[index]*dist.integrate_xfx(low = low, high = high)

        return xfx



class PiecewiseUniform(Distribution):
    """Distribution consisting of bins with probabilities"""
    def __init__(self, intervals, probabilities):
        super().__init__()
        self.intervals = intervals
        probabilities = np.array(probabilities, dtype = 'f')
        probabilities /= probabilities.sum()
        self.probabilities = probabilities
        means = []
        for inter in intervals:
            means.append((inter[0] + inter[1])/2)

        self.means = means
        self.mu = np.dot(self.means, self.probabilities)

    def __repr__(self):
        return 'PUni({:.2f})'.format(self.mu)    


    def __str__(self):
        return 'PiecewiseUniform(intervals={}, probs={})'.format(
            np.array(self.intervals), np.array(self.probabilities))

    def expectation(self):
        return self.mu

    def sample(self, n=None):
        if n is None:
            index = np.random.choice(len(self.intervals), p = self.probabilities)
            bin = self.intervals[index]
            return np.random.uniform(low = bin[0],high = bin[1])

        else:
            indices = np.random.choice(len(self.intervals), size = n, p = self.probabilities)
            samples = []
            for index in indices:
                bin = self.intervals[index]
                samples.append(np.random.uniform(low = bin[0],high = bin[1]))

            return np.array(samples) 


    def __mul__(self, other):
        if not(hasattr(other,'sample')):
            if abs(other) < 1e-10:
                return 0.0

            elif other < 0:
                intervals = []
                for inter in self.intervals:
                    intervals.append((inter[1]*other,inter[0]*other)) 
                return PiecewiseUniform(intervals, self.probabilities)    

            else:            
                intervals = []
                for inter in self.intervals:
                    intervals.append((inter[0]*other,inter[1]*other))
                return PiecewiseUniform(intervals, self.probabilities)    

    def __rmul__(self, other):
        if not(hasattr(other,'sample')):
            if abs(other) < 1e-10:
                return 0.0

            elif other < 0:
                intervals = []
                for inter in self.intervals:
                    intervals.append((inter[1]*other,inter[0]*other)) 
                return PiecewiseUniform(intervals, self.probabilities)    

            else:            
                intervals = []
                for inter in self.intervals:
                    intervals.append((inter[0]*other,inter[1]*other))
                return PiecewiseUniform(intervals, self.probabilities) 

    def __add__(self, other):        
        if hasattr(other, 'intervals'):
            #intervals = []
            #probabilities = []
            
            interdict = {}

            for ind1 in range(len(self.intervals)):
                for ind2 in range(len(other.intervals)):
                    low = self.intervals[ind1][0] + other.intervals[ind2][0]
                    high = self.intervals[ind1][1] + other.intervals[ind2][1]

                    #todo: make it faster
                    if not (low,high) in interdict:
                        #intervals.append((low,high))
                        #probabilities.append(self.probabilities[ind1]*other.probabilities[ind2])
                        interdict[(low,high)] = self.probabilities[ind1]*other.probabilities[ind2]

                    else:
                        interdict[(low,high)] += self.probabilities[ind1]*other.probabilities[ind2]                        
                        #probabilities[intervals.index((low,high))] += self.probabilities[ind1]*other.probabilities[ind2]

            intervals, probabilities = zip(*interdict.items())
            return PiecewiseUniform(list(intervals), list(probabilities))

        elif not hasattr(other, 'sample'):
            intervals = []
            for inter in self.intervals:
                intervals.append((inter[0] + other,inter[1] + other))

            return PiecewiseUniform(intervals, self.probabilities)            

    def __radd__(self, other):
        if not hasattr(other, 'sample'):
            intervals = []
            for inter in self.intervals:
                intervals.append((inter[0] + other,inter[1] + other))

            return PiecewiseUniform(intervals, self.probabilities)

    def cdf(self, k):
        cdfk = 0.0
        for index in range(len(self.intervals)):
            inter = self.intervals[index]
            if inter[0] < k:
                if inter[1] < k:
                    cdfk += self.probabilities[index]
                else:    
                    cdfk += self.probabilities[index]*(k - inter[0])/(inter[1] - inter[0])
        return cdfk

    def integrate_xfx(self, low = -np.inf, high = np.inf):
        xfx = 0.0
        for index in range(len(self.intervals)):
            inter = self.intervals[index]
            if inter[0] < high and inter[1] > low:
                lowerb = max(inter[0], low)
                upperb = min(inter[1], high)
                xfx += self.probabilities[index]*(upperb**2 - lowerb**2)/(2*(inter[1] - inter[0]))
        return xfx



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

    def variance(self):
        return self.sigma**2

    def copy(self):
        return Normal(self.mu, self.sigma)

    def sample(self, n=None):
        if n is not None:
            return self.mu + self.sigma * np.random.randn(n)
        else:
            return self.mu + self.sigma * np.random.randn()

    def sample_nocache(self):
        return self.mu + self.sigma * np.random.randn()

    def cdf(self, k):
        if self.sigma > 0:
            return norm.cdf(k, self.mu, self.sigma)
        elif self.mu < k:
            return 1.0
        return 0.0    

    def integrate_xfx(self, low = -np.inf, high = np.inf):
        mu = self.mu
        sigma = self.sigma

        if sigma > 0:
            if high == np.inf:
                alpha = (low - mu)/sigma
                return mu*(1 - standard_norm.cdf(alpha)) + sigma*standard_norm.pdf(alpha)                 
            
            alpha = (low - mu)/sigma
            beta = (high - mu)/sigma
            #return integrate.quad(lambda x: x*norm.pdf(x,mu,sigma), low, high)[0]
            return (self.cdf(high) - self.cdf(low))*truncnorm.mean(alpha, beta, loc=mu, scale=sigma)
        if low > mu or high < mu:
            return 0.0
        return mu


    @classmethod
    def fit(cls, samples):
        return cls(*scipy.stats.norm.fit(samples))
    
class TruncatedNormal(Distribution):
    """Truncated Normal distribution."""
    def __init__(self, mu, sigma, lower, upper):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.lower = lower
        self.upper = upper

        #if sigma > 0:
        #    a , b = (lower - mu)/sigma, (upper - mu)/sigma
        #    self.expect = truncnorm.mean(a ,b, loc = mu, scale = sigma)
        #else:

        #Not technically correct; assumes symmetricity about location
        self.expect = mu

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
        return self.expect

    def variance(self):
        return self.sigma**2

    def copy(self):
        return TruncatedNormal(self.mu, self.sigma, self.lower, self.upper)

    def sample(self, n=None):
        a , b = (self.lower - self.mu) / self.sigma, (self.upper - self.mu) / self.sigma
        return truncnorm.rvs(a, b, loc = self.mu, scale = self.sigma, size=n)

    def cdf(self, k):
        if self.sigma > 0:
            a , b = (self.lower - self.mu)/self.sigma, (self.upper - self.mu)/self.sigma
            return truncnorm.cdf(k, a, b, loc = self.mu, scale = self.sigma)
        elif self.mu < k:
            return 1.0
        return 0.0
    
    def integrate_xfx(self, low = -np.inf, high = np.inf):
        mu = self.mu
        sigma = self.sigma

        if sigma > 0:
            a = max(low,self.lower)
            b = min(high,self.upper)
            if a > b:
                return 0.0 
            alpha = (a - mu)/sigma
            beta = (b - mu)/sigma
            #return integrate.quad(lambda x: x*norm.pdf(x,mu,sigma), low, high)[0]
            return (self.cdf(b) - self.cdf(a))*truncnorm.mean(alpha, beta, loc=mu, scale=sigma)
        if low > mu or high < mu:
            return 0.0
        return mu

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

    def to_sampledist(self, n=10000):
        d = SampleDist(self.sample(n))
        ev = self.mu @ self.weights
        d.expectation = lambda: ev
        return d

    def expectation(self):
        return self.mu

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

def variance(val):
    try:
        return val.variance()
    except AttributeError:
        return 0


ZERO = PointMass(0)

def sample(val, n = None):
    try:
        return val.sample(n=n)
    except AttributeError:
        return val

def cross(dists, f=None):
    if f is None:
        f = lambda *x: x
    outcomes = Counter()
    for outcome_probs in it.product(*dists):
        o, p = zip(*outcome_probs)
        outcomes[f(*o)] += reduce(lambda x, y: x*y, p)

    return Categorical(outcomes.keys(), outcomes.values())



__no_default__ = 25

# @lru_cache(maxsize=None)
def cmax(dists, default=__no_default__):
    dists = tuple(dists)
    if len(dists) == 1:
        return dists[0]
    elif len(dists) == 0:
        if default is not __no_default__:
            return default
        else:
            raise ValueError('dmax() arg is an empty sequence')
    else:
        return cross(dists, max)


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
