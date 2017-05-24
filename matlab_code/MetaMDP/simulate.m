%After get_MDPpseudorewards, meta_semi_gradient_SARSA, value_guess,
%policy_compare

cost = 0.001;
for k=1:2
reward = 0;
for j=1:10000
    cs = [1,1];
    I = 1;
    r = 0;
    for i=1:30
        if compare(I,k) == 1
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
        elseif compare(I,k) == 2
            r = r + max([cs(1)/(cs(1)+cs(2)),cs(2)/(cs(1)+cs(2))]);
            break
        end
    end
    reward = reward + r;
end
er = reward/10000
end



