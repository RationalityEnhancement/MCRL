classdef BayesianRLAgent
    %RL agent that learns and decides using Bayesian value function
    %approximation.
    
    properties
        glm;
        feature_extractors;
        is_action_feature;
    end
    
    methods
        function agent=BayesianRLAgent(nr_features,feature_extractors,is_action_feature,mu,sigma)
            %features: planning horizon and whether it is equal to the total number of moves            
            if exist('sigma','var')
                agent.glm=BayesianGLM(nr_features,sigma);                        
            else
                agent.glm=BayesianGLM(nr_features);                        
            end            
            
            agent.feature_extractors=feature_extractors;
            agent.is_action_feature=logical(is_action_feature);
            
            if exist('mu','var')
                agent.glm.mu_0=mu;    
                agent.glm.mu_n=mu;
            end
            
        end
        
        function [action,agent]=decide(agent,state,mdp)
            
            %Thompson sampling for contextual bandit problems
            w_hat=agent.glm.sampleCoefficients();            
            
            [actions,~]=mdp.getActions(state);
            Q_hat=zeros(numel(actions),1);
            for a=1:numel(actions)
                features=agent.extractFeatures(mdp,state,actions(a));                
                action_features=features(agent.is_action_feature);
                Q_hat(a)=dot(w_hat(agent.is_action_feature),action_features);
            end
            a_max=argmax(Q_hat);
            action=actions(a_max);
        end
        
        
        function features=extractFeatures(agent,mdp,state,action) 
            features=agent.feature_extractors(mdp,state,action);
        end
        
        
        function [agent,discrepancy]=learn(agent,mdp,state,action,reward,next_state)
            if mdp.isTerminalState(next_state)
                value_estimate=reward;
            else
                %choose next action according to epsilon-greedy policy
                [~,mdp]=mdp.getActions(next_state);
                a_next=agent.decide(next_state,mdp);
                value_estimate=reward+mdp.gamma*dot(agent.extractFeatures(mdp,next_state,a_next)',agent.glm.mu_n);
            end
                                    
            features=agent.extractFeatures(mdp,state,action);
            discrepancy=abs(value_estimate-dot(features,agent.glm.mu_n));
            
            agent.glm=agent.glm.update(features',value_estimate);
                       
        end
    end
    
end