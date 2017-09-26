nr_states=size(lightbulb_problem(1).mdp.states,1);
pi_BO = 2*ones(nr_states,1);
X = lightbulb_problem(c).fit.features;
co =  lightbulb_problem(c).mdp.cost;

for i=1:nr_states
    vpi = X(i,1);
    voc1 = X(i,2);

    f_obs = [vpi, voc1, co];
    if f_obs*w_BO' > 0
        pi_BO(i) = 1;
    end
end

lightbulb_problem(c).BO.w_BO=w_BO;
lightbulb_problem(c).BO.pi_BO=pi_BO;
save('../results/lightbulb_problem.mat','lightbulb_problem') 