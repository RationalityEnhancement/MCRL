function [lme,SE,samples_posterior]=lmeChibJeliazkov01(data,prior,likelihood,proposal,nr_samples,evaluation_point)
%{
lmeChibJeliazkov01(prior,likelihood,proposal,nr_samples) computes the
log-model evidence log p(y|m) using the method proposed by Chib, &
Jeliazkov (2001). Marginal Likelihood from the Metropolis-Hastings Output.

evaluation_point: point in parameter space whose posterior probability density will
be estimated; high-density points lead to more accurate estimates.

prior and likelihood are structs with the following fields:
  -density: function handle returning probability density
  -log_density: function handle returning log probability density

proposal is a struct with the following fields:
  -density= @(from,to,params)
  -log_density= @(from,to,params)
  -propose= @(from,params)
  -sample= @(from,params,nr_samples)
  -params

Notes:
1. All density and log-density functions must be vectorized. If the
parameter space is d-dimensional and n is the number of samples, then the
density functions must operate on nxd-dimensional matrices.
2. Don't worry about SE being NaN. This is just because, I haven't yet
   implemented the estimate of the standard error of the log-model evidence estimate.
3. Depending on your problem and the number of samples, you may want to
increase the burn-in parameter.
%}
log_joint=@(theta) prior.log_density(theta)+likelihood.log_density(data,theta);
neg_log_joint=@(theta) -log_joint(theta);

burn_in=1000;

propose=@(from,params) proposal.propose(from,proposal.params^2);
log_proposal_prob=@(from,to,sigma) proposal.log_density(from,to,sigma);

if isfield(proposal,'initial_value')
    initial_value=proposal.initial_value;
else
    initial_value=randn(1,likelihood.nr_parameters);
end

[samples_posterior,log_joint_densities]=MetropolisHastings(log_joint,propose,initial_value,log_proposal_prob,proposal.params,nr_samples,burn_in);
samples_posterior=samples_posterior';
if not(exist('evaluation_point','var'))
    [~,MAP_sample_nr]=max(log_joint_densities); clear log_joint_densities
    evaluation_point=samples_posterior(:,MAP_sample_nr)';
end

samples_proposal=proposal.sample(evaluation_point,proposal.params,nr_samples);

logAcceptanceProb=@(from,to) max(log(eps),min(0,(likelihood.log_density(data,to)+prior.log_density(to)+proposal.log_density(to,from,proposal.params))-...
    (likelihood.log_density(data,from)+prior.log_density(from)+proposal.log_density(from,to,proposal.params))));

samples_per_block=5000;
nr_blocks=ceil(nr_samples/samples_per_block);
temp_numerator=NaN(nr_samples,1);temp_denominator=NaN(nr_samples,1);
for b=1:nr_blocks
    in_block=((b-1)*samples_per_block+1):min(b*samples_per_block,nr_samples);
    temp_numerator(in_block)=-log(nr_samples)+logAcceptanceProb(samples_posterior(:,in_block)',evaluation_point)+proposal.log_density(samples_posterior(:,in_block)',evaluation_point,proposal.params);
    temp_denominator(in_block)=-log(nr_samples)+logAcceptanceProb(evaluation_point,samples_proposal(in_block,:));
end
log_numerator=logsumexp(temp_numerator);
log_denominator=logsumexp(temp_denominator);

log_posterior_max=log_numerator-log_denominator;

lme=likelihood.log_density(data,evaluation_point)+prior.log_density(evaluation_point)-log_posterior_max;
SE=NaN; %to be computed
samples_posterior=samples_posterior';
end