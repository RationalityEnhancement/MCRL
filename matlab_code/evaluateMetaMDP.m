function [ER_hat,result]=evaluateMetaMDP(w,c)
load ../../../results/lightbulb_problem.mat
reps = 2500;
horizon = 30;
reward = 0;
X = lightbulb_problem(c).fit.features;
S = lightbulb_problem(c).mdp.states;
co =  lightbulb_problem(c).mdp.cost;
for j=1:reps
    st = [1,1];
    I = 1;
    r = 0;
    pheads = rand;
    for i=1:horizon
        %decision based on w

        I = find(S(:, 1) == st(1) & S(:, 2) == st(2));
        vpi = X(I,1);
        voc1 = X(I,2);
        
        f_obs = [vpi, voc1];
        
        if f_obs*w > 0
            r = r - co;
            flip = rand;
%                         pheads = cs(1)/(cs(1)+cs(2));
            heads = flip <= pheads;
            if heads
                st = [st(1)+ 1,st(2)];
            else
                st = [st(1),st(2) + 1];
            end
        else
            r = r + max(st)/sum(st) - min(st)/sum(st);
            break
        end
    end
    reward = reward + r;
end
ER_hat = reward/reps;
result.features={'VPI','VOC','E[R|act,b]'};
result.cost_per_click=co;
end