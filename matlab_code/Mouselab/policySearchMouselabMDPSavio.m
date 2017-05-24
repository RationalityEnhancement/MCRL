function policySearchMouselabMDPSavio(c)

addpath('/global/home/users/flieder/matlab_code/')
addpath('/global/home/users/flieder/matlab_code/MatlabTools/')

%% Direct Policy Search
nr_episodes=1000;
ER_hat=@(w) evaluatePolicy([w(:);1],c,nr_episodes);
d=2;

x_input_domain = [-1 1; -1 1];
nb_iter=25;
result_display=true; result_save=true; plot_func=false; plot_point=false;

GPO_path='/global/home/users/flieder/matlab_code/MatlabTools/Gaussian_Optimization/'
addpath(genpath(GPO_path))
cd(GPO_path)
[x, fx, X_sample, F_sample, result] = ...
    IMGPO_default_run_stochastic_objective(ER_hat, d, x_input_domain, nb_iter, ...
    result_display, result_save, plot_func, plot_point)

BO.w_hat=[x(:);1];
BO.ER=fx;
BO.cost=c;
BO.nb_iter=nb_iter;
BO.nr_episodes=nr_episodes;
save(['/global/home/users/flieder/results/BO_c',int2str(100*c),'n',int2str(nb_iter),'.mat'],'BO')
end