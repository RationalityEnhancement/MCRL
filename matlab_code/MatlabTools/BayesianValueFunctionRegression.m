function [glm_Q,MSE,R_total]=BayesianValueFunctionRegression(mdp,feature_extractor,nr_episodes,glm_policy)
%Bayesian regression is used to estimate the value function of the policy
%defined by Thompson sampling with the supplied GLM.
%inputs:
%  1. mdp: object of whose class implements the interface MDP
%  2. feature_extractor: a function that returns the state x features matrix
%  of state features when given a vector of states as its input
%  3. nr_episodes: number of training episodes
%  4. epsilon: probability that the epsilon-greedy policy will choose an
%  action at random

%outputs:
%  1. glm: GLM with posterior distribution on the feature weights
%  2. avg_MSE: average mean-squared error in the prediction of the state
%  value by training episode.


[s0,mdp0]=mdp.newEpisode();
actions=mdp0.actions;


mu=zeros(size(feature_extractor(s0,actions(1),mdp)));
nr_features=length(mu);
glm_Q=BayesianGLM(nr_features,0.1);


avg_MSE=zeros(nr_episodes,1);

R_total=zeros(nr_episodes,1);
nr_observations=0;
for i=1:nr_episodes
    rewards=zeros(0,1);
    F=zeros(0,nr_features);
    
   [s,mdp]=mdp.randomStart();
    
    t=0; %time step within episode

    while not(mdp.isTerminalState(s))
        t=t+1;
                
        %1. Choose action by Thompson sampling
        action=contextualThompsonSampling(s,mdp,glm_policy);
        features=feature_extractor(s,action,mdp)';
        F=[F;features];
        
        %2. Observe outcome
        [r,s,PR]=mdp.simulateTransition(s,action);
        rewards=[rewards;r];
        
        R_total(i)=R_total(i)+r;
        nr_observations=nr_observations+1;
                
        %value_estimate=dotglm_Q.mu_n*features
        %PE=value_estimate-dot(glm_Q.mu_n,features);
        
        %avg_MSE(i)=((t-1)*avg_MSE(i)+PE^2)/t;
        
        if any(or(isnan(glm_Q.mu_n),isinf(glm_Q.mu_n)))
            throw(MException('MException:isNaN','MSE is NaN'))
        end
    end
    
    nr_actions_in_episode=size(rewards,1);
    returns=zeros(nr_actions_in_episode,1);
    for a=1:nr_actions_in_episode
        returns(a,1)=sum(rewards(a:end));
    end
    predicted_returns=F*glm_Q.mu_n;
    MSE(i)=norm(predicted_returns-returns)^2/nr_actions_in_episode;
    
    glm_Q=glm_Q.update(F,returns);
    
    if mod(i,250)==0
        disp(['Completed episode ',int2str(i)])
    end
    %disp(['MSE=',num2str(avg_MSE(i)),', |mu_n|=',num2str(norm(glm.mu_n)),', return: ',num2str(R_total(i))])
end

end