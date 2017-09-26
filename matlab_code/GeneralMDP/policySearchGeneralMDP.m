function [x, fx, X_sample, F_sample, result] = policySearchGeneralMDP(nr_arms, nr_balls)
work_dir=cd;
toolbox_dir=[work_dir,'/MatlabTools/'];
GPO_dir = [toolbox_dir,'/Gaussian_Optimization/'];
addpath(work_dir)
addpath(toolbox_dir)
addpath(genpath(GPO_dir))
% Direct Policy Search
ER_hat=@(w) evaluateGeneralMDP([w(:)],nr_arms,nr_balls);

x_input_domain = [-1 1; -1 1;-1 1; -1 1];
nb_iter=550;
result_display=true; result_save=true; plot_func=false; plot_point=false;
d=4;

cd(GPO_dir)
[x, fx, X_sample, F_sample, result] = ...
    IMGPO_default_run_stochastic_objective(ER_hat, d, x_input_domain, nb_iter, ...
    result_display, result_save, plot_func, plot_point);
end