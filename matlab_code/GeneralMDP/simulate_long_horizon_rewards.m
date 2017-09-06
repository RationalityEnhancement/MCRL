reps = 100;
eps = 2000;
num_arms = 2:6;
mu = zeros(numel(num_arms),reps);
v = zeros(numel(num_arms),reps);
begin = 0.005;
endin = 0.01;
% costs = begin:(endin-begin)/20:endin;
costs = 0.007;
for c=1:numel(costs)
    cost = costs(c)
for n=1:numel(num_arms)
    n_arms = num_arms(n);
    er = zeros(eps,reps/10);
for i=1:eps %each sequence of observations
    st = ones(n_arms,2);
    accum = 0;
    for j=1:reps %observe each arm for many steps
        for k=1:n_arms
            accum = accum-cost;
            pheads = st(k,1)/sum(st(k,:));
            if rand <= pheads
                st(k,1) = st(k,1) +1;
            else
                st(k,2) = st(k,2) +1;
            end
        end
        er(i,j) = max(st(:,1)./sum(st,2)) + accum;
    end
end

for i=1:reps
    mu(n,i) = mean(er(:,i));
    v(n,i) = 1.96/sqrt(eps)*std(er(:,i));
end
disp(argmax(mu(n,:)));
end
graph_ex_rewards
end