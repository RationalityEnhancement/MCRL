classdef PlanExecuter
    %UNTITLED8 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        plan;
        has_plan=false;
        step_nr=0;
        horizon;
        mdp
        cost_of_planning=0.25;
    end
    
    methods
        
        function system2=PlanExecuter(mdp)
            system2.mdp=mdp;            
        end
        
        
        function [action,agent]=decide(agent,state,mdp)
            
                agent.step_nr=agent.step_nr+1;                
                if agent.has_plan
                    action=agent.plan(state,agent.step_nr);            
                else
                    %no plan --> choose randomly
                    available_actions=mdp.getActions(state);
                    action=drawSample(available_actions);
                end
        end
        
        %This is a placeholder that does nothing.
        function [agent,discrepancy]=learn(agent,mdp,s,a,r,s_next)
            discrepancy=0;
        end
                
    end
    
end