function [T,R,states]=oneArmedMouselabMDP(nr_hallway_cells,nr_branches,nr_leafs_per_branch,mu_reward,sigma_reward,cost)

nr_leafs = nr_branches*nr_leafs_per_branch;
nr_cells = nr_hallway_cells + nr_leafs;

hallway_cells = 1:nr_hallway_cells;

first_leaf=nr_hallway_cells+1;
leafs= first_leaf:nr_cells;

single_leafs=leafs(1:nr_leafs/2);
all_leafs=setdiff(leafs,single_leafs);

getSiblings = @(leaf) leaf+(1-2*mod(leaf-first_leaf,nr_leafs_per_branch));

sigma0=sqrt(nr_cells)*sigma_reward;

[E_max,std_max]=EVofMaxOfGaussians([mu_reward;mu_reward],[sigma_reward;sigma_reward]);

resolution=sigma0/10;
mu_values=(mu_reward-2*sigma0):resolution:(mu_reward+2*sigma0);
sigma_values=sigma_reward*(0:0.5:2*nr_cells);
[MUs,SIGMAs]=meshgrid(mu_values,sigma_values);

sample_values=(mu_reward-3*sigma0):resolution:(mu_reward+3*sigma0);
for s=1:numel(sample_values)    
    [E_max_given_x1(s),STD_max_given_x1(s)]=EVofMaxOfGaussians(...
        [sample_values(s);mu_reward],[0,sigma_reward]);
end


observation_indices = 0:(power(2,nr_cells)-1);
nr_observation_indices = numel(observation_indices);

observation_vectors=zeros(nr_cells,nr_observation_indices);
for o_id=1:nr_observation_indices
    observation_vectors(:,o_id)=str2num(dec2bin(observation_indices(o_id),nr_cells)');
end


states.mu=repmat(MUs,[1,1,nr_observation_indices]);
states.sigma=repmat(SIGMAs,[1,1,nr_observation_indices]);
for id=1:nr_observation_indices
    states.observation_id(:,:,id)=1+observation_indices(id)*ones(size(MUs));
end
states.nr_observation_indices=nr_observation_indices;

%state = (obs_id, delta_mu, sigma_mu)
nr_states=numel(MUs)*2^nr_cells+1; %each combination of mu and sigma is a state and there is one additional terminal state
nr_actions=nr_cells+1; %action 0 = act, action i: observe cell i

state_nr = @(observation_vector,mu,sigma) (bi2de(observation_vector)+1)+...
    nr_observation_indices*(find(and( MUs(:)==mu,SIGMAs(:)==sigma))-1);

%b) define transition matrix
T=zeros(nr_states,nr_states,nr_actions);
R=zeros(nr_states,nr_states,nr_actions);
%The first action (c_0) terminates deliberation and takes action. It
%therefore transitions into the the terminal state.
T(:,:,1)=repmat([zeros(1,nr_states-1),1],[nr_states,1]);

R(:,:,2:nr_actions)=-cost; %cost of sampling

for from=1:(nr_states-1)
    
    from_state.mu=states.mu(from);
    from_state.sigma=states.sigma(from);
    from_state.obs_id=states.observation_id(from);
    from_state.observed=observation_vectors(:,states.observation_id(from));
    
    if from_state.sigma>0 %there is still something to be observed        
        
        for a=2:nr_actions
            
            inspected_cell=a-1;
            to.observed=min(1,from_state.observed'+deltaDistribution(a-1,1:nr_cells));
            to.obs_id=bi2de(to.observed)+1;

            %compute new state according to which cell has been inspected
            if ismember(inspected_cell,hallway_cells)
                %a hallway state has been inspected
                sample_values=(mu_reward-3*from_state.sigma):resolution:(mu_reward+3*from_state.sigma);
                p_samples=discreteNormalPMF(sample_values,mu_reward,from_state.sigma);
                
                posterior_means  = (from_state.mu + sample_values - mu_reward );
                posterior_sigmas = sqrt(from_state.sigma^2-sigma_reward^2)*ones(size(posterior_means));
            elseif ismember(inspected_cell,single_leafs)
                %this action inspects only one of the leafs at the
                %exclusion of inspecting its siblings
                twig = [inspected_cell,getSiblings(inspected_cell)];
                
                if any(from_state.observed(twig))
                    %This action is unavailble if one or more of the leafs
                    %have already been observed. So there is no change in
                    %that case.
                    posterior_means = from_state.mu;
                    posterior_sigmas = from_state.sigma;
                    p_samples=1;
                else
                    %Sample the value of one leaf and update the mean
                    %and variance of the twig's value according to the sampled value.
                    sample_values=(mu_reward-3*sigma0):resolution:(mu_reward+3*sigma0);
                    p_samples=discreteNormalPMF(sample_values,mu_reward,from_state.sigma);
                    
                    posterior_means  = (from_state.mu + E_max_given_x1 - mu_reward );
                    posterior_sigmas = sqrt(from_state.sigma^2-std_max^2+STD_max_given_x1.^2);
                end
                
            elseif ismember(inspected_cell,all_leafs)
                %inspecting all leafs of a twig
                
                twig = [inspected_cell,getSiblings(inspected_cell)];
                
                if any(from_state.observed(twig))
                    %This action is unavailble if one or more of the leafs
                    %have already been observed. So there is no change in
                    %that case.
                    posterior_means = from_state.mu;
                    posterior_sigmas = from_state.sigma;
                    p_samples=1;
                else
                    %Sample from the distribution of the maximum and update
                    %the value of the tree accordingly.

                    sample_values=(mu_reward-3*from_state.sigma):resolution:(mu_reward+5*from_state.sigma);
                    pdf_samples=pdfOfMaxOfGaussians(sample_values,[mu_reward;mu_reward],[sigma_reward;sigma_reward]);
                    p_samples=pdf_samples*resolution; %probability mass function
                    
                    posterior_means  = (from_state.mu + sample_values - mu_reward );
                    posterior_sigmas = sqrt(max(0,from_state.sigma^2-std_max^2))*ones(size(posterior_means));
                end
            end
            
            [discrepancy_mu, mu_index] = min(abs(repmat(posterior_means,[numel(mu_values),1])-...
                repmat(mu_values',[1,numel(posterior_means)])));
            
            [discrepancy_sigma, sigma_index] = min(abs(repmat(posterior_sigmas,[numel(sigma_values),1])-...
                repmat(sigma_values',[1,numel(posterior_sigmas)])));
                        
            to.mu=mu_values(mu_index);
            to.sigma=sigma_values(sigma_index);
            to.nr=zeros(numel(to.mu),1);
            for n=1:numel(to.mu)
                to.nr(n)=state_nr(to.observed,to.mu(n),to.sigma(n));
            end
            
            %sum the probabilities of all samples that lead to the same state
            if numel(to.nr)==1
                T(from,to.nr,a)=1;
            else
                T(from,unique(to.nr),a)=grpstats(p_samples(:),to.nr(:),{@sum});
            end
        end
        
    else
        T(from,:,2:end)=repmat([zeros(1,nr_states-1),1],[1,1,nr_actions-1]);
        R(from,end,2:end)=-cost;
    end
    
    %reward of acting
    R(from,nr_states,1)=max(0,from_state.mu);
end
T(:,:,1)=repmat([zeros(1,nr_states-1),1],[nr_states,1]); %bot transitions into the terminal state
T(end,:,:)=repmat([zeros(1,nr_states-1),1],[1,1,nr_actions]);

%{
states.MUs=MUs;
states.SIGMAs=SIGMAs;
states.start_state=start_state;
states.mu_values=mu_values;
states.sigma_values=sigma_values;
%}

end