function no_planning_problems=worthless_planning()

load medium_cost_condition

load baseline_mdp
nr_states = numel(baseline_mdp.states);
baseline_mdp.actions=1:4;
baseline_mdp.nextState=medium_cost_condition(1).nextState;

paths = [1 2 6 10; 1 2 6 11; 1 3 7 12; 1 3 7 13; 1 4 8 14; 1 4 8 15; ...
    1 5 9 16; 1 5 9 17];
actions = [1 1 2; 1 1 4; 2 2 3; 2 2 1; 3 3 2; 3 3 4; 4 4 3; 4 4 1];

nr_paths = size(paths,1);
nr_moves = size(paths,2)-1;
nr_branches = nr_paths/2;
min_return = 20; max_return = 35;

nr_trials = 16;

for t = 1:nr_trials
    
    no_planning_problems(t)=baseline_mdp;
    
    total_return(t)=min_return+randi(max_return-min_return);
    for b=1:nr_branches
        
        rewards(1:nr_moves-1) = randi(total_return(t),nr_moves-1,1);
        rewards(nr_moves) = total_return(t) - sum(rewards(1:nr_moves-1));
        rewards=shuffle(rewards);
        
        for p=(2*b-1):2*b
            for m=1:nr_moves
                from = paths(p,m);
                to = paths(p,m+1);
                action = actions(p,m);
                no_planning_problems(t).rewards(from,to,action) = rewards(m);
            end
        end
        if mod(t,2)==0
            no_planning_problems(t).rewards(from,to,action) = rewards(m)+0.5;
        else
            no_planning_problems(t).rewards(from,to,action) = rewards(m)-0.5;
        end

    end
end

end