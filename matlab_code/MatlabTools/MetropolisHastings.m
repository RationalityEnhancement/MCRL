function [samples,log_joint_densities]=MetropolisHastings(log_joint,propose,initial_value,log_proposal_prob,proposal_params,nr_samples,burn_in)
%Metropolis-Hastings Algorithm
%log_proposal_prob(from,to)

if not(exist('burn_in','var'))
    burn_in=0;
end

samples(1,:)=initial_value;
log_joint_densities(1)=log_joint(initial_value);

nr_accepted=0;
for s=1:(nr_samples+burn_in-1)
    proposal=propose(samples(s,:),proposal_params);
        
    log_joint_proposal=log_joint(proposal);
    
    if isinf(-log_joint_proposal)
        acceptance_probability=0;
    else
        log_q_to_current=log_proposal_prob(proposal,samples(s,:),proposal_params);
        log_q_to_proposal=log_proposal_prob(samples(s,:),proposal,proposal_params);
        
        prob_ratio=exp(log_joint_proposal-log_joint_densities(s)+log_q_to_current-log_q_to_proposal);
        if isnan(prob_ratio)
            throw(MException('ResultChk:NaN','probability ratio is not a number.'))
        end
        
        acceptance_probability=min(1,prob_ratio);
    end
    
    if rand()<acceptance_probability
        samples(s+1,:)=proposal;
        if s>burn_in
            nr_accepted=nr_accepted+1;
        end
        log_joint_densities(s+1)=log_joint_proposal;
    else
        samples(s+1,:)=samples(s,:);
        log_joint_densities(s+1)=log_joint_densities(s);
    end
end

samples=samples(burn_in+1:end,:);
log_joint_densities=log_joint_densities(burn_in+1:end);
acceptance_rate=nr_accepted/nr_samples;
disp(['The acceptance rate was ',num2str(acceptance_rate)]);
end