function mdps=generateIIDMDPs(baseline_mdp,mu_reward,std_reward,nr_values,nr_problems)


nr_moves = size(baseline_mdp.rewards,3);

mdps=repmat(baseline_mdp, [nr_problems,1]);

payoff_values = round(linspace(mu_reward - 2 * std_reward, mu_reward+2*std_reward, nr_values));
delta = payoff_values(2)-payoff_values(1);
p_payoffs(1,1)=normcdf(payoff_values(1)+delta/2,mu_reward,std_reward);
for v=2:length(payoff_values)-1
    p_payoffs(v,1)=normcdf(payoff_values(v)+delta/2,mu_reward,std_reward)-...
        normcdf(payoff_values(v-1)+delta/2,mu_reward,std_reward);
end
p_payoffs(nr_values,1)=1-normcdf(payoff_values(end)-delta/2,mu_reward,std_reward);

nr_states = size(baseline_mdp.T,1);
for rep = 1:nr_problems
    
    for m=1:nr_moves
        sampled_rewards = sampleDiscreteDistributions(p_payoffs',nr_states^2,payoff_values);
        mdps(rep).rewards(:,:,m)=mdps(rep).T(:,:,m).* reshape(sampled_rewards, [nr_states,nr_states]);
    end
end

end