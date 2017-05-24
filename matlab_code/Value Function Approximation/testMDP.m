classdef testMDP < MDP
    
    properties
        action_features=3;
    end
    
    methods
        function mdp=testMDP(nr_actions,gamma)
            mdp.nr_actions=nr_actions;
            mdp.actions = 1:nr_actions;
            mdp.gamma=gamma;
            mdp.actions=1:mdp.nr_actions;
            mdp.action_features=1;
        end
        
        function [s,mdp]=sampleS0(mdp)
            s=0;
        end
        
        function [s0,mdp]=newEpisode(mdp)
            mdp=testMDP(mdp.nr_actions,mdp.gamma);
            s0=mdp.sampleS0();
        end
        
        function true_or_false=isTerminalState(mdp,s)
            true_or_false=s==1;
        end
        
        function ER=expectedReward(mdp,s,a)
            if s==0
                ER=1;
            else
                ER=0;
            end
        end
        
        function [r,s_next,PR]=simulateTransition(mdp,s,a)
                        
            if s==0
                s_next=1;
                r=1;
            else
                r=0;
                s_next=2;
            end
            
            PR=0;
        end
        
        function [next_states,p_next_states]=predictNextState(mdp,s,a)
            next_states=[0,1,2];
            
            if s==0
                p_next_states=[0,1,0];
            elseif s==1
                p_next_states=[0,0,1];
            end
                
        end
        
        function [actions,mdp]=getActions(mdp,state)
            actions=1:mdp.nr_actions;
        end
        
        function [action_features]=extractActionFeatures(mdp,state,action)
            action_features=[action];
        end
                        
    end

end