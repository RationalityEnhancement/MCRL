function policySearchMouselabMDP(c,experiment,name,fast_VOC1_approximation)

username = getenv('USER');
Falk_local=strcmp(username,'Falk');

if Falk_local    
    root_dir='~/Dropbox/PhD/Metacognitive RL/MCRL/';    
else
    root_dir='/global/home/users/flieder/';    
end

work_dir=[root_dir,'matlab_code/'];
toolbox_dir=[work_dir,'MatlabTools/'];
GPO_dir = [toolbox_dir,'Gaussian_Optimization/'];
addpath(work_dir)
addpath(toolbox_dir)
addpath(genpath(GPO_dir))

%% Direct Policy Search

if not(exist('fast_VOC1_approximation','var'))
    fast_VOC1_approximation = true;
end

if exist('experiment','var')
    nr_episodes=numel(experiment);
    ER_hat=@(w) evaluatePolicy([w(:);1],c,nr_episodes,experiment,fast_VOC1_approximation);
else
    nr_episodes=1000;
    ER_hat=@(w) evaluatePolicy([w(:);1],c,nr_episodes,fast_VOC1_approximation);
end

d=4;

x_input_domain = [-1 2; -1 1; -1 1; -5 1];
nb_iter=100;
%nb_iter=50;
result_display=true; result_save=true; plot_func=false; plot_point=false;

cd(GPO_dir)
[x, fx, X_sample, F_sample, result] = ...
    IMGPO_default_run_stochastic_objective(ER_hat, d, x_input_domain, nb_iter, ...
    result_display, result_save, plot_func, plot_point);

BO.w_hat=[x(:);1];
BO.ER=fx;
BO.cost=c;
BO.nb_iter=nb_iter;
BO.nr_episodes=nr_episodes;

if exist('name','var')
    save([root_dir,'results/BO/BO_c',int2str(100*c),'n',int2str(nb_iter),name,'.mat'],'BO')
else
    save([root_dir,'results/BO/BO_c',int2str(100*c),'n',int2str(nb_iter),'.mat'],'BO')
end

end