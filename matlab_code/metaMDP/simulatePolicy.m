function er = simulatePolicy(st_init,a_init)
    load ../results/lightbulb_problem.mat 
    if a_init == 2 || sum(st_init) >= 31 || st_init(1) == -1
        er = (max(st_init)-min(st_init))/sum(st_init); 
    else
        c = 7;
        reps = 10;
        S = lightbulb_problem(c).mdp.states;
        policy = lightbulb_problem(c).BO.pi_BO;
        cost = lightbulb_problem(c).mdp.cost;
        reward = -cost;
        
        for j=1:reps
            %first toss
            cs = st_init;
            r = 0;
            flip = rand;
            pheads = cs(1)/(cs(1)+cs(2));
            heads = flip <= pheads;
            if heads
                cs = [cs(1)+ 1,cs(2)];
            else
                cs = [cs(1),cs(2) + 1];
            end
            I = find(S(:, 1) == cs(1) & S(:, 2) == cs(2));
            %remaining actions
             
            for i=1:30
                if policy(I) == 1
                    r = r - cost;
                    flip = rand;
                    pheads = cs(1)/(cs(1)+cs(2));
                    heads = flip <= pheads;
                    if heads
                        cs = [cs(1)+ 1,cs(2)];
                    else
                        cs = [cs(1),cs(2) + 1];
                    end
                    I = find(S(:, 1) == cs(1) & S(:, 2) == cs(2));
                elseif policy(I) == 2
                    r = r + (max(cs) - min(cs))/sum(cs);
                    break
                end
            end
            reward = reward + r;
        end
        er = reward/reps;
    end
end