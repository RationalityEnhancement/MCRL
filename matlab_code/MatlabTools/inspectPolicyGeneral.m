function [R_total,problems,states,chosen_actions,indices]=...
    inspectPolicyGeneral(mdp,policy,nr_episodes)
%simulates the acting according to policy on the environment mdp.
%inputs:
%  1. mdp: object of whose class implements the interface MDP
%  2. policy: a mapping from state and mdp to action: action=policy(state,mdp) 
%  3. nr_episodes: number of training episodes

%outputs:

[~,mdp0]=mdp.newEpisode();
actions=mdp0.actions;

R_total=zeros(nr_episodes,1);
nr_observations=0;
for i=1:nr_episodes

    [s,mdp]=mdp.newEpisode();
    problems(i)=mdp;
        
    t=0; %time step within episode
    prev_choice=[NaN,NaN];
    while not(mdp.isTerminalState(s))
        t=t+1;
        
        states{i,t}=s;
                
        %1. Choose action
        [actions,mdp]=mdp.getActions(s);
        actions=policy(s,mdp);
        
        for a=1:numel(actions)
            action=actions(a);
            chosen_actions{i}(t)=action;
            
            if action.is_decision_mechanism
                [state,decision]=mdp.decide(s,action);
                chosen_actions{i}(t).move=decision;
                chosen_actions{i}(t).from_state=state.s;
            end
            
            %2. Observe outcome
            [r,s_next,PR]=mdp.simulateTransition(s,action);
            if action.is_decision_mechanism
                chosen_actions{i}(t).state=s_next.s;
            end
            
            R_total(i)=R_total(i)+r;
            nr_observations=nr_observations+1;
            
            s=s_next;
        end
    end
    
    %EV(choice)/[max_g EV(g)]
    %EVs=mdp.payoff_matrix'*mdp.outcome_probabilities(:);
    %max_EV(i)=max(mdp.payoff_matrix'*mdp.outcome_probabilities(:));
    %EV_of_choice(i)=EVs(action.gamble);
    %indices.percent_optimal_EV(i)=100*EV_of_choice(i)/max_EV(i);
    
    %analyze difference between compensatory vs. non-compensatory problems
    %indices.was_compensatory(i)=max(mdp.outcome_probabilities)<=0.40;
    
    
    %number of acquisitions
    indices.nr_acquisitions(i)=sum(not(isnan(s.observations(:))));
    
    %proportion of time spent on the Most Important Attribute (MIA)
    %MIA=argmax(mdp.outcome_probabilities);
    %indices.PTPROB(i)=sum(not(isnan(s.observations(MIA,:))))/sum(not(isnan(s.observations(:))));
        
    %variance across attributes
    %nr_acquisitions_by_attribute=sum(not(isnan(s.observations)),2);
    %indices.var_attribute(i)=var(nr_acquisitions_by_attribute);
    
    %variance accross alternatives
    %nr_acquisitions_by_alternative=sum(not(isnan(s.observations)),1);
    %indices.var_alternative(i)=var(nr_acquisitions_by_alternative);
    
    %relative frequency of alternative-based processing
    %indices.pattern(i)=(nr_alternative_based_transitions-nr_attribute_based_transitions)/...
    %    (nr_alternative_based_transitions+nr_attribute_based_transitions);
    
    %disp(['MSE=',num2str(avg_MSE(i)),', |w|=',num2str(norm(w)),', return: ',num2str(R_total(i))])
end

%indices.effect_of_dispersion.means=[nanmean(indices.PTPROB(~indices.was_compensatory)),
%    nanmean(indices.PTPROB(indices.was_compensatory))];
%[h,p,ci,stats]=ttest2(indices.PTPROB(~indices.was_compensatory),indices.PTPROB(indices.was_compensatory));
%indices.effect_of_dispersion.stats=stats;
%indices.effect_of_dispersion.p=p;

end