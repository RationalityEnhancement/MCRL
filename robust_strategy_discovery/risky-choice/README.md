# RSD for Mouselab Risky Choice

Methodology followed: 
1. Formulate true distributions
2. Run experiment to capture people's biases (with Mateo Tosic)
3. Model biases (with Yash Raj Jain) and arrive at posterior distributions of true reward structure
4. Utilize them in RL algorithms and compare to their vanilla counterparts
5. Run experiment to teach discovered strategies to people and test them on the true distributions in step 1 (with Yash Raj Jain)

Note: The 2 experiments (Steps 2 & 5) are in their respective repositories.

## About the reward distributions
* General distribution structure: 
    * Rewards sampled from range (-1, 1). 
    * Distribution approximated using 10 bins of equal width, where the probability within a particular bin is uniform. 
    * Implemented here as a class called PiecewiseUniform. 
* Prior: 
    * 10 dimensional Dirichlet distribution, modelling the probability of each bin. Calculated in dirichlet_prior.ipynb.
* Likelihood: 
    * Multinomial over the output of the bias model.
* Posterior:
    * Analytically difficult, so Metropolis-Hastings used to sample from the posterior directly. This is currently done in each algorithm file separately (where required).


## About the strategy-discovery algorithms
There are implementations for 4 algorithms - BMPS, DRQN, Meta-RL and MVOC:
* `*_biased.py`: training on participants' biased descriptions of the environment.
* `drqn.py` and `metarl.py`: training on samples from the posterior
* `bmps_posterior-mhmcmc` and `mvoc_posterior-mhmcmc`: Approximate posterior using a dirichlet distribution, then use it.
* `bmps_posterior_updates-mhmcmc`: Same as above but also updates expectation in-trial based on the rewards already observed.


## Misc.
* benchmarks.ipynb - Heuristics people are known to use in the environment.
* bmps_nn_for_attributes, bmps_true : NN implementation of bmps features. Only added for completeness, can be ignored.
* stock_dist_sandbox.ipynb - Trying out various "true" distributions on which people's biases are collected.

## Resources
* [Paper](https://www.researchgate.net/publication/232454399_Adaptive_Strategy_Selection_in_Decision_Making) that introduces the Mouselab environment
* [Paper](https://cocosci.princeton.edu/papers/Meta_Decision_Making-CameraReady.pdf) that talks about strategy-discovery in this paradigm
* [BMPS](https://arxiv.org/abs/1711.06892)
* [DRQN](https://arxiv.org/abs/1507.06527)
* [Meta-RL](https://arxiv.org/abs/1611.05763)
