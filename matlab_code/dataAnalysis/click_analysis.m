nr_trials = 16;
dat = data;
% %% Separate data into 3 sets if necessary
% low_cost = 0.01;
% med_cost = 1;
% high_cost = 2.5;
% 
% low_data = data([data.info_cost1] == low_cost);
% med_data = data([data.info_cost1] == med_cost);
% high_data = data([data.info_cost1] == high_cost);
% 
% %% Change the dataset you're working with here
% dat = high_data;
% trial_properties = trial_properties_high;
%% Click inner until positive, then try that arm
target = [2, 3, 4, 5];
relevant_trials = 0;

for k=1:length(dat)
    all_trials = dat(k).clicks1;
    for t = 1:nr_trials
        cur_trial = dat(k).trialID(t) + 1;
        rewards = trial_properties(cur_trial).reward_by_state;
        clicks = all_trials{1, t};
        if length(clicks) < 2
            continue
        end
        if ~ismember(clicks(1), target) %check that first click is one of the inner four
            continue
        else
            for c = 2:length(clicks)
                click = clicks(c);
                previous_click = clicks(c-1);
                prev_click_val = rewards(previous_click);
                if ~ismember(click, target) % check if a click has been made outside the inner circle
                    if prev_click_val >= 0
                        if previous_click == 2
                            arm = [6, 10, 11];
                        elseif previous_click == 3
                            arm = [7, 12, 13];
                        elseif previous_click == 4
                            arm = [8, 14, 15];
                        elseif previous_click == 5
                            arm = [9, 16, 17];
                        end
                        if ismember(click, arm) 
                            relevant_trials = relevant_trials + 1;
                        end
                        break
                    else
                        break %break because the previous click had a negative value
                    end
                else % this means the participant clicked in the inner four again
                    if prev_click_val >= 0
                        break %break because participant already found a positive value, but continued to click in the inner four
                    end
                end
            end
        end   
    end
end

percentage_of_inner_til_positive = (relevant_trials/(length(dat)* nr_trials)) * 100;

%% Click inner 4, then middle of best
target = [2, 3, 4, 5];
stop_at_five = 0;
best_leaves = 0;
still_best_leaves = 0;
next_best = 0;
second_middle = 0;
anywhere = 0;
other = 0;

for k=1:length(dat)
    all_trials = dat(k).clicks1;
    for t = 1:nr_trials
        if length(all_trials{1, t}) < 5
            continue
        end
        cur_trial = dat(k).trialID(t) + 1;
        clicks = all_trials{1, t}(1:4);
        match = all(ismember(clicks, target));
        if match == 1
            rewards = trial_properties(cur_trial).reward_by_state;
            inner_rewards = trial_properties(cur_trial).reward_by_state(2:5);
            [max_value, node_index] = max(inner_rewards);
            node_index = node_index + 1;
            if node_index == 2
                target_set = [6];
                leaves = [10, 11];
            elseif node_index == 3
                target_set = [7];
                leaves = [12, 13];
            elseif node_index == 4
                target_set = [8];
                leaves = [14, 15];
            elseif node_index == 5
                target_set = [9];
                leaves = [16, 17];
            end
            fifth_click = all_trials{1, t}(5);
            if ismember(fifth_click, target_set) 
                fifth_value = rewards(fifth_click);
                if (fifth_value > -1) && (length(all_trials{1, t}) > 5)
                    sixth_click = all_trials{1, t}(6);
                    if ismember(sixth_click, leaves)
                        best_leaves = best_leaves + 1;
                    else
                        other = other + 1;
                    end
                elseif length(all_trials{1, t}) > 5
                    sixth_click = all_trials{1, t}(6);
                    if ismember(sixth_click, leaves)
                        still_best_leaves = still_best_leaves + 1;
                    end
                    inner_rewards(node_index - 1) = -60; %setting the highest value to the lowest, so we can find the next highest value
                    [max_value, node_index] = max(inner_rewards);
                    node_index = node_index + 1;
                    if node_index == 2
                        target_set = [6, 10, 11];
                        leaves = [10, 11];
                    elseif node_index == 3
                        target_set = [7, 12, 13];
                        leaves = [12, 13];
                    elseif node_index == 4
                        target_set = [8, 14, 15];
                        leaves = [14, 15];
                    elseif node_index == 5
                        target_set = [9, 16, 17];
                        leaves = [16, 17];
                    end
                    if ismember(sixth_click, target_set)
                        next_best = next_best + 1; % a click in the arm adjacent to the second best node
                        if sixth_click == min(target_set)
                            second_middle = second_middle + 1; %the node adjacent to the second best node
                        end
                    else
                        anywhere = anywhere + 1; % a click in the leaves of any other arm but second best
                    end
                else
                    stop_at_five = stop_at_five + 1; % p made a move/decision after only five clicks
                end
            end
        end
    end
end

percentage_of_five_clicks = (stop_at_five/(length(dat) * nr_trials)) * 100;
percentage_of_best_leaves = (best_leaves/(length(dat) * nr_trials)) * 100;
percentage_of_still_best_leaves = (still_best_leaves/(length(dat) * nr_trials)) * 100;
percentage_of_next_best = (next_best/(length(dat) * nr_trials)) * 100;
percentage_of_anywhere = (anywhere/(length(dat) * nr_trials)) * 100;
percentage_of_other = (other/(length(dat) * nr_trials)) * 100;
percentage_of_next_middle = (second_middle/(length(dat) * nr_trials)) * 100;

clearvars k all_trials t cur_trial clicks match max_value node_index target_set leaves stop_at_five best_leaves next_best anywhere other fifth_click sixth_click

%% Click leaf nodes until postive value, then go
target = [10, 11, 12, 13, 14, 15];
%target = [6, 7, 8, 9];
relevant_trials = 0;

for k=1:length(dat)
    all_trials = dat(k).clicks1;
    for t = 1:nr_trials
        cur_trial = dat(k).trialID(t) + 1;
        rewards = trial_properties(cur_trial).reward_by_state;
        clicks = all_trials{1, t};
        for c = 1:length(clicks)
            click = clicks(c);
            if ismember(click, target)
                click_value = rewards(click);
                if click_value >= 0 && (c == length(clicks))
                    relevant_trials = relevant_trials + 1;
                end
            else
                break
            end
        end
    end
end

leaf_and_go_percentage = (relevant_trials/(length(dat) * nr_trials)) * 100;


%% Trying out an entire "arm" in first 4 clicks
% Change 4 to 3 to see how many only tried 3 locations in an arm
upper = [3, 7, 12, 13];
right = [2, 6, 10, 11];
bottom = [5, 9, 16, 17];
left = [4, 8, 14, 15];

relevant_trials = 0;

for k=1:length(dat)
    all_trials = dat(k).clicks1;
    for t = 1:nr_trials
        if length(all_trials{1, t}) < 4 
            continue
        end
        clicks = all_trials{1, t}(1:4);
        up_match = all(ismember(clicks, upper));
        r_match = all(ismember(clicks, right));
        bottom_match = all(ismember(clicks, bottom));
        l_match = all(ismember(clicks, left));
        if up_match || r_match || bottom_match || l_match
            relevant_trials = relevant_trials + 1;
        end
    end
end

arm_percentage = (relevant_trials/(length(dat) * nr_trials)) * 100;

clearvars k all_trials t cur_trial clicks match node_index fifth_click up_match r_match bottom_match l_match upper right bottom left
%% Trying out partial arms
upper = [3, 7, 12, 13];
right = [2, 6, 10, 11];
bottom = [5, 9, 16, 17];
left = [4, 8, 14, 15];

relevant_trials = 0;
few_clicks = 0;
skipped_trials = 0;

for k=1:length(dat)
    all_trials = dat(k).clicks;
    for t = 1:nr_trials
        current_trial = dat(k).trialID(t, 1) + 1;
        all_clicks = str2num(all_trials{t, 1}); 
        if length(all_clicks) < 4 
            skipped_trials = skipped_trials + 1; % This skips about 30% of the trials (15% don't click at all, other 15% have between 1 and 3 clicks)
            continue
        end
        clicks = all_clicks(1:3);
        up_match = all(ismember(clicks, upper));
        r_match = all(ismember(clicks, right));
        bottom_match = all(ismember(clicks, bottom));
        l_match = all(ismember(clicks, left));
        if up_match || r_match || bottom_match || l_match
            %Find the value of the leaf node inspected and whether there
            %was a move made after it 
            loc = clicks(3);
            leaf_val = trial_properties(current_trial).reward_by_state(loc);
            if leaf_val > 10
                if length(all_clicks) == 3 %I hypothesize that it's unlikely to terminate after 3 clicks but
                    few_clicks = few_clicks + 1;
                end
            elseif leaf_val < 10
                next_try = all_clicks(4); %which arm was picked next?
                %see if the next 2 clicks also belong to the same arm
                %if the value was high enough, see if termination happened
                %or keep going
            end
        end
    end
end
percentage_after_three = (few_clicks/(length(dat) * nr_trials)) * 100;

clearvars k all_trials t cur_trial clicks match node_index fifth_click up_match r_match bottom_match l_match upper right bottom left

%% Percentage of no clicks vs all clicks
num_none = 0; %number of trials where nothing was clicked
num_all = 0; %number of trials where everything was clicked
num_one = 0; %number of trials with just one click
for k=1:length(n_click)
    num_clicks = n_click(k, 1);
    if num_clicks == 0
        num_none = num_none + 1;
    elseif num_clicks == 16
        num_all = num_all + 1;
    elseif num_clicks == 1
        num_one = num_one + 1;
    end
end

percentage_of_no_clicks = (num_none/(length(dat) * nr_trials)) * 100;
percentage_of_all_clicks = (num_all/(length(dat) * nr_trials)) * 100;
percentage_of_one_click = (num_one/(length(dat) * nr_trials)) * 100;
        