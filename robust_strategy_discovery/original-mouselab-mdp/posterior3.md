The environment is the following:
- Level 1: Only nodes of type 'L'
- Level 2: Only nodes of type 'M'
- Level 3: Nodes of type 'M' and 'H' with equal probability of occurrence. The node types per branch are the same.

Therefore, the theta space consists of 8 environments.

These are the rewards for each node type (although this information is not necessary for producing the posteriors):
- 'L': [-4.0, -2.0, 2.0, 4.0],
- 'M': [-8.0, -4.0, 4.0, 8.0],
- 'H': [-48.0, -24.0, 24.0, 48.0]

The modeled bias represents imperfect memory and inclination to more extreme values. There are two probabilities of confusion. The numbers are arbitrary and can be changed.
- Given that the node type in the true environment is 'M', the person memorizes the truth ('M') 45% of the time and has bias towards more extreme values ('H') 55% of the time.
- Given that the node type in the true environment is 'H', the person memorizes the truth ('H') 90% of the time and has imperfect memory ('M') 10% of the time.

Therefore, the theta_hat space is the same as the theta space.

The accuracy of the model based on 10000 samples is about 30%. A random classifier would produce 12.5%.

The plots can be interpreted in the following way: If the biased environment is X (name/title of the plot), the person thinks that the original environment is Y with probability <bar>).