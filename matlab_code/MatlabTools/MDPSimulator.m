classdef MDPSimulator
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        states
        actions
        transition_probabilities
        reward_function
        pseudorewards;
        initial_distribution
        terminal_states
        gamma=1
        remaining_nr_steps;
        nr_steps;
        actions_by_state;
    end
    
    methods
        function mdp=MDPSimulator(states,actions,T,reward_function,P0,terminal_states,pseudorewards,horizon,actions_by_state)
            mdp.states=states;
            mdp.actions=actions;
            mdp.transition_probabilities=T;
            mdp.reward_function=reward_function;
            mdp.initial_distribution=P0;
            mdp.terminal_states=terminal_states;
            
            if exist('actions_by_state','var')
                mdp.actions_by_state=actions_by_state;               
            else
                mdp.actions_by_state=repmat({actions},[numel(states),1]);
            end
            
            if exist('pseudorewards','var')
                mdp.pseudorewards=pseudorewards;
            else
                mdp.pseudorewards=zeros(size(reward_function));
            end
            
            if exist('horizon','var')
                mdp.nr_steps=horizon;
                mdp.remaining_nr_steps=horizon;
            else
                mdp.nr_steps=inf;
                mdp.remaining_nr_steps=inf;                
            end
        end
        
        function [agents,R_total,episodes,avg_discrepancy,total_PR]=simulate(mdp,agent0,nr_episodes,nr_simulations,horizon,learn_from_PR)
            
            if not(exist('horizon','var'))
                horizon=mdp.nr_steps;
            end            
            
            if not(exist('learn_from_PR','var'))
                learn_from_PR=false;
            end               
            
            R_total.mean=NaN(nr_episodes,1);
            R_total.sem=NaN(nr_episodes,1);
            R_total.returns=NaN(nr_simulations,nr_episodes);
            episodes.states=cell(nr_simulations,nr_episodes);
            episodes.actions=cell(nr_simulations,nr_episodes);
            episodes.rewards=cell(nr_simulations,nr_episodes);
            
            for sim=1:nr_simulations
                agent=agent0;
                
                for e=1:nr_episodes
                    state_seq=[mdp.sampleS0()];
                    reward_seq=[];
                    PR_seq=[];
                    action_seq=[];
                    
                    step=1;
                    while and(not(mdp.isTerminalState(state_seq(step))),step<=horizon)
                        [action_seq(step),agent]=agent.decide(state_seq(step),mdp);
                        [reward_seq(step),state_seq(step+1),PR_seq(step)]=...
                            mdp.simulateTransition(state_seq(step),action_seq(step));
                        
                        mdp.remaining_nr_steps=mdp.remaining_nr_steps-1;
                        
                        if learn_from_PR
                            [agent,discrepancy(step)]=agent.learn(mdp,state_seq(step),action_seq(step),...
                                PR_seq(step),state_seq(step+1));
                        else
                            [agent,discrepancy(step)]=agent.learn(mdp,state_seq(step),action_seq(step),...
                                reward_seq(step)+PR_seq(step),state_seq(step+1));
                        end
                        step=step+1;
                    end
                    episodes.rewards{sim,e}=reward_seq;
                    episodes.pseudorewards{sim,e}=PR_seq;
                    episodes.actions{sim,e}=action_seq;
                    episodes.states{sim,e}=state_seq;
                    R_total.returns(sim,e)=sum(reward_seq); 
                    avg_discrepancy(sim,e)=mean(discrepancy);
                    PR_total.returns(sim,e)=sum(PR_seq);
                end
                
                agents{sim}=agent;
            end
            R_total.mean=mean(R_total.returns);
            R_total.sem=sem(R_total.returns);
            PR_total.mean=mean(PR_total.returns);
            PR_total.sem=sem(PR_total.returns);
            
        end
        
        function s0=sampleS0(mdp)
            s0=sampleDiscreteDistributions(mdp.initial_distribution',1,mdp.states);
        end
        
        function is_final=isTerminalState(mdp,state)
            is_final=ismember(state,mdp.terminal_states);
        end
        
        function [reward,next_state,pseudoreward]=simulateTransition(mdp,state,action)
            next_state=sampleDiscreteDistributions(mdp.transition_probabilities(state,:,action));
            reward=mdp.reward_function(state,next_state,action);
            pseudoreward=mdp.pseudorewards(state,next_state);
        end
    
        function [actions,mdp]=getActions(mdp,state)
            actions=mdp.actions_by_state{state};
        end
    end
    
end

