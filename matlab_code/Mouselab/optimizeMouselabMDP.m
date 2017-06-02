function [current_MDP,current_score,optimality,properties]=...
    optimizeMouselabMDP(mdp,nr_iterations,reward_range,myopic_first_move,...
    optimal_planning_horizon)

if not(exist('optimal_planning_horizon','var'))
    optimal_planning_horizon=mdp.horizon;
end

if not(exist('myopic_first_move','var'))
    myopic_first_move=false;
end

possible_transitions=find(mdp.T(:)>0);

current_MDP=mdp;
properties=evaluateMouselabMDP(current_MDP.T,current_MDP.rewards,...
    current_MDP.start_state,current_MDP.horizon,myopic_first_move,optimal_planning_horizon);
current_score=properties.score;

if not(exist('reward_range','var'))
    reward_range=50;
end

proposal_range=max(2,round(2*reward_range/10));
for i=1:nr_iterations
    
    
    %propose change
    %flag=true;
    %while (flag)
    change=randi(proposal_range+1,size(possible_transitions))-(proposal_range/2+1);
    proposed_MDP=current_MDP;
    proposed_MDP.rewards(possible_transitions)=proposed_MDP.rewards(possible_transitions)+change;
    
    scaling_factor=reward_range/range(proposed_MDP.rewards(possible_transitions));    
    proposed_MDP.rewards(possible_transitions)=roundToMultipleOf(...
        scaling_factor*proposed_MDP.rewards(possible_transitions),1);
    
        
    %    flag=or(max(proposed_MDP.rewards(:))>r_max,min(proposed_MDP.rewards(:))<-r_min);
    %end
    
    %score proposed MDP
    properties_new=evaluateMouselabMDP(proposed_MDP.T,proposed_MDP.rewards,proposed_MDP.start_state,proposed_MDP.horizon,myopic_first_move,optimal_planning_horizon);
    proposal_score=properties_new.score;
    
    %accept if it improves or the score
    if proposal_score>current_score
        current_MDP=proposed_MDP;
        current_score=proposal_score;        
    elseif proposal_score<=current_score
        p_accept=0.5*(proposal_score+0.1)/(current_score+0.1);
        
        if rand()<p_accept
            current_MDP=proposed_MDP;
            current_score=proposal_score;
            properties=properties_new;
        end
    end
    %accept probabilistically otherwise
    
    if properties.optimality>=1.6
        break
    end
end

optimality=properties.optimality;

end