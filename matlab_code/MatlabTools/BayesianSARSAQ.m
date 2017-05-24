function [glm,avg_MSE,R_total]=BayesianSARSAQ(mdp,feature_extractor,nr_episodes,glm)
%semi-gradient RL algorithm that learns a linear feature-based
%approximation to the state value function for the epsilon-greedy policy.
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

if not(exist('glm','var'))
    mu=zeros(size(feature_extractor(s0,actions(1),mdp)));
    nr_features=length(mu);
    glm=BayesianGLM(nr_features,0.1);
end


avg_MSE=zeros(nr_episodes,1);

R_total=zeros(nr_episodes,1);
nr_observations=0;
for i=1:nr_episodes

    [s,mdp]=mdp.newEpisode();
    
    t=0; %time step within episode
    a_next=contextualThompsonSampling(s,mdp,glm);
    features_next=feature_extractor(s,a_next,mdp);
    while not(mdp.isTerminalState(s))
        t=t+1;
                
        %1. Choose action by Thompson sampling
        action=a_next;
        features=features_next;
        
        %2. Observe outcome
        [r,s_next,PR]=mdp.simulateTransition(s,action);
        R_total(i)=R_total(i)+r;
        nr_observations=nr_observations+1;
                
        if mdp.isTerminalState(s_next)
            value_estimate=r+PR;
        else
            %choose next action by contextual Thompson sampling
            [~,mdp]=mdp.getActions(s_next);
            a_next=contextualThompsonSampling(s_next,mdp,glm);
            next_features=feature_extractor(s_next,a_next,mdp);
            value_estimate=r+PR+mdp.gamma*dot(next_features,glm.mu_n);
        end
        
        PE=value_estimate-dot(glm.mu_n,features);
                
        glm=glm.update(features',value_estimate);
        
        s=s_next;
        
        avg_MSE(i)=((t-1)*avg_MSE(i)+PE^2)/t;
        
        if any(or(isnan(glm.mu_n),isinf(glm.mu_n)))
            throw(MException('MException:isNaN','MSE is NaN'))
        end
    end
    
    if mod(i,250)==0
        disp(['Completed episode ',int2str(i)])
    end
    %disp(['MSE=',num2str(avg_MSE(i)),', |mu_n|=',num2str(norm(glm.mu_n)),', return: ',num2str(R_total(i))])
end

end