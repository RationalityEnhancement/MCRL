%After get_MDPpseudorewards, meta_semi_gradient_SARSA, value_guess,
%policy_compare
% costs=logspace(-3,-1/4,15);
reps = 3000;
results.er = zeros(numel(costs),5);
results.sterr = zeros(numel(costs),5);
results.samples = zeros(numel(costs),5,reps);
states = lightbulb_problem(1).mdp.states;
S = states;
nr_states = size(states,1);
pols = zeros(numel(costs),nr_states,5);

for c=1:numel(costs)
    cost = costs(c);
    [m1, pols(c,:,1)] = max(lightbulb_problem(c).mdp.Q_star,[],2);
    [m2, pols(c,:,2)] = max(lightbulb_problem(c).BSARSA.Q_hat_BSARSA,[],2);
    [m3, pols(c,:,3)] = max(lightbulb_problem(c).fit.Q_hat,[],2);
    pols(c,:,4) = lightbulb_problem(c).pi_meta;
    pols(c,:,5) = lightbulb_problem(c).BO.pi_BO;
    
    for k=1:5
        reward = 0;
        rewards = zeros(reps,1);
        for j=1:reps
            cs = [1,1];
            I = 1;
            r = 0;
%             pheads = rand;
            for i=1:30
                if pols(c,I,k) == 1
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
                elseif pols(c,I,k) == 2
                    r = r + max([cs(1)/(cs(1)+cs(2)),cs(2)/(cs(1)+cs(2))]) - min([cs(1)/(cs(1)+cs(2)),cs(2)/(cs(1)+cs(2))]);
                    break
                end
            end
            reward = reward + r;
            rewards(j) = r;
            results.samples(c,k,j) = r;
        end
        results.er(c,k) = reward/reps;
        results.sterr(c,k) = std(rewards);
    end
end

generateMetaMDPfigures

save('../results/results.mat','results') 