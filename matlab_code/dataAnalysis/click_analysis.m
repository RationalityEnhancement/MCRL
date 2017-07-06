nr_trials = 12;
low_cost = 0.01;
med_cost = 1;
high_cost = 2.5;

low_data = data([data.info_cost1] == low_cost);
med_data = data([data.info_cost1] == med_cost);
high_data = data([data.info_cost1] == high_cost);

dat = med_data;
trial_properties = trial_properties_med;
%% Click inner 4, then any of best
target = [2, 3, 4, 5];
relevant_trials = 0;

for k=1:length(dat)
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
            if node_index == 2
                target_set = [6, 10, 11];
            elseif node_index == 3
                target_set = [7, 12, 13];
            elseif node_index == 4
                target_set = [8, 14, 15];
            elseif node_index == 5
                target_set = [9, 16, 17];
            end
            fifth_click = all_trials{1, t}(5);
            if ismember(fifth_click, target_set) 
                relevant_trials = relevant_trials + 1;
            end
        end
    end
end

percentage_of_fifth_click = (relevant_trials/(length(dat) * nr_trials) * 100);

clearvars k all_trials t cur_trial clicks match node_index fifth_click



