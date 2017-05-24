function [T,R,states]=GaussianSamplingMetaMDP(start_state,cost)

sigma0=start_state.sigma;

resolution=sigma0/50;
delta_mu_values=-2*sigma0:resolution:2*sigma0;
sigma_values=sigma0:-resolution:0.1;
[MUs,SIGMAs]=meshgrid(delta_mu_values,sigma_values);
nr_states=numel(MUs)+1; %each combination of mu and sigma is a state and there is one additional terminal state
nr_actions=2; %action 1: sample, action 2: act

%b) define transition matrix
T=zeros(nr_states,nr_states,nr_actions);
R=zeros(nr_states,nr_states,nr_actions);

R(:,:,1)=-cost; %cost of sampling

for from=1:(nr_states-1)
        current_mu=MUs(from);
        current_sigma=SIGMAs(from);
        sample_values=(current_mu-3*current_sigma):resolution:(current_mu+3*current_sigma);
        p_samples=discreteNormalPMF(sample_values,current_mu,current_sigma);
        
        %In this case, the prior is the likelihood. Hence, both have the
        %same precision. Therefore, the posterior mean is the average of
        %the prior mean and the observation, and the posterior precision is
        %twice as high as the current precision.
        
        posterior_means  = (current_mu + sample_values)/2;
        posterior_sigmas = repmat(1/sqrt(2*1/current_sigma^2),size(posterior_means));
        
        [discrepancy_mu, mu_index] = min(abs(repmat(posterior_means,[numel(delta_mu_values),1])-...
            repmat(delta_mu_values',[1,numel(posterior_means)])));

        [discrepancy_sigma, sigma_index] = min(abs(repmat(posterior_sigmas,[numel(sigma_values),1])-...
            repmat(sigma_values',[1,numel(posterior_sigmas)])));
        
        to=struct('mu',delta_mu_values(mu_index),'sigma',sigma_values(sigma_index),...
        'index',sub2ind([numel(sigma_values),numel(delta_mu_values)],sigma_index,mu_index));
        
        %sum the probabilities of all samples that lead to the same state
        T(from,unique(to.index),1)=grpstats(p_samples,to.index,{@sum});
        
        %reward of acting
        R(from,nr_states,2)=max(0,current_mu);
end
T(:,:,2)=repmat([zeros(1,nr_states-1),1],[nr_states,1]);
T(end,:,:)=repmat([zeros(1,nr_states-1),1],[1,1,2]);

start_state.index=sub2ind(size(MUs),find(sigma_values==start_state.sigma),...
    find(delta_mu_values==start_state.delta_mu));

states.MUs=MUs;
states.SIGMAs=SIGMAs;
states.start_state=start_state;
states.delta_mu_values=delta_mu_values;
states.sigma_values=sigma_values;
end