function pseudoR_matrix = get_pseudoreward_matrix(S,values,min_trial,discount,R)
% pseudoR_matrix = get_pseudoreward_matrix(S,values,min_trial,discount)
% 
% this function returns a (s x (n+1)) pseudoreward matrix, where s is
% the number of states and n is the number of trials. Note that every state
% has some minimum number of trials needed to reach it; for this reason, 
% pseudoR_matrix will be a quasi-upper-diagonal matrix.  See
% get_MDPpseudorewards.m for details
% 
% values: s x (n+1) matrix of state values, where s=number of states,
% n=number of trials
% 
% S: s x 2 matrix showing the number of heads (coloumn 1) and the number of
% tails (column 2) in each state (rows)
% 
% min_trial: s x 1 vector of the minimum possible trial number for a given
% state; i.e. if the agent only samples, what trial number will it be in
% for each state
% 
% discount: scalar value of discount on future rewards


pseudoR_matrix1 = nan(size(values));

second_to_last_state = find(sum(S,2)==max(sum(S,2))-1,1,'last');
% we only go to the second-to-last state because we don't have values for
% for states beyond {[1,100],[2,99],...,[100,1]}, which we would need for
% calculating pseudorewards.
for i = 1:size(values,1) %second_to_last_state % loop over states
    
    next_state1 = find(S(:,1)==(S(i,1)+1) & S(:,2)==S(i,2)); % if "heads"
    next_state2 = find(S(:,1)==S(i,1) & S(:,2)==(S(i,2)+1)); % if "tails"
    prob_next_state1 = S(i,1)/sum(S(i,:)); % probability of sampling heads
    prob_next_state2 = S(i,2)/sum(S(i,:)); % probability of sampling tails

    expected_reward = R(i,2);
    
    % loop over trials-- only trials that are possible for a given state
    for j = 1:size(values,2)-1 %min_trial(i):size(values,2)-1

        value_this_state = values(i,j);
        if i > second_to_last_state
            value_next_state1 = 0;
            value_next_state2 = 0;
        else
            value_next_state1 = values(next_state1,j+1);
            value_next_state2 = values(next_state2,j+1);
        end
        expected_value_sample = value_next_state1*prob_next_state1 + ...
            value_next_state2*prob_next_state2;
        pseudoR_matrix1(i,j) = discount*expected_value_sample - value_this_state;% + expected_reward;
    end
    
end

pseudoR_matrix2 = nan(size(values));

% action 2 (bet) pseudorewards
for i = 1:size(values,1) %second_to_last_state % loop over states
    
    next_state = i;
    
    expected_reward = R(i,2);
    
    % loop over trials-- only trials that are possible for a given state
    for j = 1:size(values,2)-1 %min_trial(i):size(values,2)-1

        value_this_state = values(i,j);
        if i > second_to_last_state
            value_next_state = 0;
        else
            value_next_state = values(next_state,j+1);
        end                
        
        pseudoR_matrix2(i,j) = discount*value_next_state - value_this_state;% + expected_reward;
    end
    
end

pseudoR_matrix1 = pseudoR_matrix1;
pseudoR_matrix2 = pseudoR_matrix2;

pseudoR_matrix = cat(3,pseudoR_matrix1,pseudoR_matrix2);