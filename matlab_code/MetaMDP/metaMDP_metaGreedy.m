nr_states=size(lightbulb_problem(1).mdp.states,1);
S=lightbulb_problem(1).mdp.states;
pi_meta = zeros(10,nr_states);

for c=1:numel(costs)
    cost = costs(c);
    voc1 = zeros(nr_states,2);
    
    for i=1:nr_states
        st = S(i,:);
         
        t = sum(st);
        mvoc = 1/(t*(t+1))*(st(1)*(max(st(1)+1,st(2))-min(st(1)+1,st(2))) + st(2)*(max(st(1),st(2)+1)-min(st(1),st(2)+1)));
        voc1(i,1) = mvoc-max(st)/sum(st)-cost+min(st)/sum(st);
        voc1(i,2) = 0;
    end
    [m,pi_meta(c,:)] = max(voc1,[],2);
    
    lightbulb_problem(c).pi_meta = pi_meta(c,:);
end
save('../results/lightbulb_problem.mat','lightbulb_problem')