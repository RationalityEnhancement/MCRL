pi_BO = zeros(nr_states,1);
X = nlightbulb_problem.fit.features;
co =  nlightbulb_problem.mdp.cost;

for i=1:nr_states
    f_obs = X(nr_arms*(i-1)+(1:nr_arms),1:2);
    [m,idx] = max(f_obs*w_BO');
    if m > 0
        pi_BO(i) = idx;
    else
        pi_BO(i) = nr_arms+1;
    end
end

nlightbulb_problem.BO.w_BO=w_BO;
nlightbulb_problem.BO.pi_BO=pi_BO;
save(['../results/', num2str(nr_arms),'lightbulb_problem.mat'],'nlightbulb_problem');