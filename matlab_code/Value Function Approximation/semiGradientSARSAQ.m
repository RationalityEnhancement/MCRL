function [w,avg_MSE,R_total]=semiGradientSARSAQ(mdp,feature_extractor,nr_episodes,alpha0,w0)
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
%  1. w: learned vector of feature weights
%  2. avg_MSE: average mean-squared error in the prediction of the state
%  value by training episode.

if not(exist('alpha0','var'))
    alpha0=1;
end


[s0,mdp0]=mdp.newEpisode();
actions=mdp0.actions;

if exist('w0','var')
    w=w0;
else
    fe = feature_extractor(s0',actions(1),mdp);
    w=zeros(size(fe));
end
avg_MSE=zeros(nr_episodes,1);

R_total=zeros(nr_episodes,1);
nr_observations=0;
for i=1:nr_episodes

    [s,mdp]=mdp.newEpisode();
    
    t=0; %time step within episode
    while not(mdp.isTerminalState(s))
        t=t+1;
                
        epsilon=1/log(nr_observations+2);
        
        %1. Choose action
        [actions,mdp]=mdp.getActions(s);
        action=epsilonGreedyPolicy(s,mdp.actions,...
                @(s,a) subindex(feature_extractor(s',a,mdp),mdp.action_features),...
                w(mdp.action_features),epsilon);
        
        %2. Observe outcome
        [r,s_next,PR]=mdp.simulateTransition(s,action);
        R_total(i)=R_total(i)+r;
        nr_observations=nr_observations+1;
        
        %3. Update weights
        alpha=alpha0/sqrt(nr_observations+2);
        fe = feature_extractor(s',action,mdp);
        prediction=dot(fe,w);
        if isinf(prediction)
            throw(MException('MException:isNaN','prediction is inf'))
        end
        
        if mdp.isTerminalState(s_next)
            value_estimate=r+PR;
        else
            %choose next action according to epsilon-greedy policy
            [actions,mdp]=mdp.getActions(s);
            a_next=epsilonGreedyPolicy(s_next,actions,...
                @(s,a) subindex(feature_extractor(s',a,mdp),mdp.action_features),...
                w(mdp.action_features),epsilon);
            value_estimate=r+PR+mdp.gamma*dot(feature_extractor(s_next',a_next,mdp),w);
        end
        features=feature_extractor(s',action,mdp);
        PE=prediction-value_estimate;
        w=w+alpha*(value_estimate-dot(features,w))*features;
        
        s=s_next;
        
        avg_MSE(i)=((t-1)*avg_MSE(i)+PE^2)/t;
        
        if any(or(isnan(w),isinf(w)))
            throw(MException('MException:isNaN','MSE is NaN'))
        end
    end
    
    disp(['MSE=',num2str(avg_MSE(i)),', |w|=',num2str(norm(w)),', return: ',num2str(R_total(i))])
end