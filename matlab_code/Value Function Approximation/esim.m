[m1,pol] = max(qh,[],2);
reps = 1000;
cost = costs(c);
reward = 0;

for j=1:reps
    cs = [1,1];
    I = 1;
    r = 0;
    pheads = rand;
    for i=1:30

            if pol == 1
                r = r - cost;
                flip = rand;
%                         pheads = cs(1)/(cs(1)+cs(2));
                heads = flip <= pheads;
                if heads
                    cs = [cs(1)+ 1,cs(2)];
                else
                    cs = [cs(1),cs(2) + 1];
                end
                I = find(S(:, 1) == cs(1) & S(:, 2) == cs(2));
            elseif pol == 2
                r = r + max([cs(1)/(cs(1)+cs(2)),cs(2)/(cs(1)+cs(2))]) - min([cs(1)/(cs(1)+cs(2)),cs(2)/(cs(1)+cs(2))]);
                break
            end
    end
    reward = reward + r;
end
er = reward/reps;