dat = data;
nr_trials = 12;
target = [2, 3, 4, 5];
cost = 2.5;
%% Find the percentage of trials in which the first 4 clicks are equivalent to the inner 4 locations
relevant_trials = 0;
same_cost_trials = 0;
for k = 1:length(dat)
    if dat(k).info_cost1 == cost
        same_cost_trials = same_cost_trials + 1;
        all_trials = dat(k).clicks1;
        for t = 1:nr_trials
            if length(all_trials{1, t}) < 4 
                continue
            end
            clicks = all_trials{1, t}(1:4);
            match = all(ismember(clicks, target));
            if match == 1
                relevant_trials = relevant_trials + 1;
            end
        end
    end
end

percentage_of_inner_four = (relevant_trials/(same_cost_trials * nr_trials) * 100);

clearvars k t clicks match all_trials

%% percentage of trials where 5th click is location adjacent to most profitable inner node
relevant_trials = 0;
if cost == 0.1
    trial_properties = trial_properties_low;
elseif cost == 1
    trial_properties = trial_properties_med;
elseif cost == 2.5
    trial_properties = trial_properties_high;
end
     
for k = 1:length(dat)
    if dat(k).info_cost1 == cost
        all_trials = dat(k).clicks1;
        for t = 1:nr_trials
            if length(all_trials{1, t}) < 4 || length(all_trials{1, t}) < 5
                continue
            end
            cur_trial = dat(k).trialID(t) + 1;
            clicks = all_trials{1, t}(1:4);
            match = all(ismember(clicks, target));
            if match == 1
                rewards = trial_properties(cur_trial).reward_by_state(2:5);
                [max_value, node_index] = max(rewards);
                node_index = node_index + 1;
                fifth_click = all_trials{1, t}(5);
                if fifth_click == (node_index + 4) 
                    relevant_trials = relevant_trials + 1;
                end
            end
        end
    end
end

percentage_of_fifth_click = (relevant_trials/(same_cost_trials * nr_trials) * 100);

clearvars k all_trials t cur_trial clicks match node_index fifth_click