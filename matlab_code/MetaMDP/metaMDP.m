classdef metaMDP < MDP
    
    properties
        action_features = [];
        rewardCorrect = 1;
        rewardIncorrect = 0;
        cost = 0.001;
        discount = 1;
        
    end
    methods
        function mdp=metaMDP(nr_actions,gamma,nr_features,cost)
            
            if not(exist('cost','var'))
                cost=0.001;
            end
            
            mdp.nr_actions=nr_actions;
            mdp.actions = 1:nr_actions;
            mdp.gamma=gamma;
            mdp.cost=cost;
            mdp.actions=1:mdp.nr_actions;
            mdp.action_features=1:nr_features;
        end
        
        function [s,mdp]=sampleS0(mdp)
            s=[1,1];
        end
        
        function [s0,mdp]=newEpisode(mdp)
            %mdp=metaMDP(mdp.nr_actions,mdp.gamma);
            s0=mdp.sampleS0();
        end
        
        function true_or_false=isTerminalState(mdp,s)
            true_or_false=s(1)+s(2)>30 || s(1) == -1;
        end
        
        function ER=expectedReward(mdp,s,a)
            if a == 1
                ER = -mdp.cost;
            elseif a == 2
                ER = mdp.rewardCorrect*max([s(1)/(s(1)+s(2)),s(2)/(s(1)+s(2))]);
            end
        end
        
        function [r,s_next,PR]=simulateTransition(mdp,s,a)
            flip = rand;
            pheads = s(1)/(s(1)+s(2));
            heads = flip <= pheads;
            if s == [-1,-1]
                r = 0;
                s_next = [-1,-1];
            elseif a == 1
                r = -mdp.cost;
                if heads
                    s_next = [s(1)+ 1,s(2) ];
                else
                    s_next = [s(1),s(2) + 1];
                end
            elseif a == 2
                %{
                if heads
                    if s(1) > s(2)
                        r = mdp.rewardCorrect;
                    else
                        r = 0;
                    end
                else
                    if s(2) > s(1)
                        r = mdp.rewardCorrect;
                    else
                        r = 0;
                    end
                end
                %}
                
                r = max(s)/sum(s);
                
                s_next = [-1,-1];
            end 
            PR = 0;
        end
        
        function [next_states,p_next_states]=predictNextState(mdp,s,a)
            next_states=[[s(1)+1,s(2)],[s(1),s(2)+1],[-1,s(2)]];
            if s(1) == -1
                p_next_states=1; 
                next_states=[-1,-1];
            elseif a == 2
                p_next_states=[0,0,1];
            elseif a == 1 
                p = s(1)/(s(1)+s(2));
                p_next_states=[p,1-p,0];
            end
                
        end
        
        function [actions,mdp]=getActions(mdp,s)
            actions=1:mdp.nr_actions;
        end
        
        function [action_features]=extractActionFeatures(mdp,state,action)
            action_features=feature_extractor(state,action,mdp);
        end
                
    end

end