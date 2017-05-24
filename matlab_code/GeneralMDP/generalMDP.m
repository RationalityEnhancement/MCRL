classdef generalMDP < MDP
    
    properties
        action_features = [];
        rewardCorrect = 1;
        rewardIncorrect = 0;
        cost=0.001;
        discount = 1;
        h;
        nr_features;
        nr_arms;
    end
    methods
        function mdp=generalMDP(nr_arms,gamma,nr_features,cost,h)
            
            if not(exist('cost','var'))
                cost=0.001;
            end
            mdp.h = h;
            mdp.nr_arms=nr_arms;
            mdp.cost = cost;
            mdp.nr_actions=nr_arms+1; 
            mdp.actions = 1:nr_arms+1;
            mdp.gamma=gamma;
            mdp.nr_features = nr_features;
            mdp.actions=1:mdp.nr_actions;
            mdp.action_features= 1:nr_features;
        end
        
        function [s,mdp]=sampleS0(mdp)
            s=ones(mdp.nr_arms,2);
        end
        
        function [s0,mdp]=newEpisode(mdp)
            mdp=generalMDP(mdp.nr_arms,mdp.gamma,mdp.nr_features,mdp.cost,mdp.h);
            s0=mdp.sampleS0();
        end
        
        function true_or_false=isTerminalState(mdp,s)
            true_or_false=sum(s(:))>2*mdp.nr_arms+mdp.h || s(1) == -1;
        end
        
        function ER=expectedReward(mdp,s,a)
            if a <= mdp.nr_arms
                ER = -mdp.cost;
            elseif a == mdp.nr_arms+1
                ER = mdp.rewardCorrect*max(s(:,1)./sum(s,2));
            end
        end
        
        function [r,s_next,PR]=simulateTransition(mdp,s,a)
            flip = rand;
            if s(1) == -1
                r = 0;
                s_next = -ones(mdp.nr_arms,2);
            elseif a <= mdp.nr_arms
                pheads = s(a,1)/(s(a,1)+s(a,2));
                heads = flip <= pheads;
                r = -mdp.cost;
                s_next = s;
                if heads
                    s_next(a,:)=[s_next(a,1)+ 1,s_next(a,2)];
                else
                    s_next(a,:)=[s_next(a,1),s_next(a,2)+1];
                end
            elseif a == mdp.nr_arms+1
                pheads = max(s(:,1));
                heads = flip <= pheads;
                if heads
                    r = mdp.rewardCorrect;
                else
                    r = 0;
                end
                s_next = -ones(mdp.nr_arms,2);
            end 
            PR = 0;
        end
        
        function [next_states,p_next_states]=predictNextState(mdp,s,a)
            if s(1) == -1
                p_next_states=1; 
                next_states=-ones(mdp.nr_arms,2);
            elseif a == mdp.nr_arms
                p_next_states=1; 
                next_states=-ones(mdp.nr_arms,2);
            elseif a <= mdp.nr_arms
                next_states=[s,s,-ones(mdp.nr_arms,2)];
                next_states(1,a,1) = s(a,1)+1;
                next_states(1,a,2) = s(a,2)+1;
                p = s(a,1)/(s(a,1)+s(a,2));
                p_next_states=[p,1-p,0];
            end
                
        end
        
        function [actions,mdp]=getActions(mdp,s)
            actions=1:mdp.nr_actions;
        end
        
        function [action_features]=extractActionFeatures(mdp,state)
            action_features=feature_extractor(state,action,mdp,selected_features);
        end
                
    end

end