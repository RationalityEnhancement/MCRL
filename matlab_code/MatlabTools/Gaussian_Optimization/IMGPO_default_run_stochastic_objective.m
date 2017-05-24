function [x, fx, X_sample, F_sample, result] = ...
                     IMGPO_default_run_stochastic_objective(f, d, x_input_domain, nb_iter, ...
                     result_diplay, result_save, plot_func, plot_point)
% execute IMGPO with a default setting used in a NIPS paper
% output: 
%      x = global optimizer 
%      fx = global optimal value f(x)
% optinal output:
%      X_sample = sampled points 
%      F_sample = sampled values of f 
%      result = intermidiate results 
%               for each iteration t, result(t,:) = [N, n (split #), fmax_hat, rho_bar, xi_max, depth_T, time(s)] 
% input: 
%      f = ovjective function (to be optimized)
%      d = input dimention of ovjective functio f
%      x_input_domain = input domain; 
%          e.g., = [-1 3; -3 3] means that domain(f) = {(x1,x2) : -1 <= x1 <= 3 and -3 <= x2 <= 3]} 
%      nb_iter = the number of iterations to perform
% input display flag:
%      result_diplay = 1: print intermidiate results
%      result_save = 1: save intermidiate result and return as result
%      plot_func = 1: plot objective function if the dimensionality is <= 2
%      plot_point = 1; plot data points if the dimensionality is <= 2

%% parameter setting 
% ------- parameter for the main algorithm ------- 
% for low dimension
XI_max = 2^2;     % to limit the computational time due to GP: 2^2 or 2^3 is computationally reasonable (see the NIPS paper for more detail)
%XI_max = 2^3;    % lower this if it is too slow 

% for higher dimension
%XI_max = 1;      
%XI_max = 2;

% ------- parameters for GP ------- 
GP_use = 1;        % = 1: use GP
nu = 0.05;         % theoretical gurantee holds with probability 1 - nu 
GP_varsigma = @(M) sqrt(2*log(pi^2*M^2/(12*nu)));  % UCB = mean + GP_varsigma(M) * sigma

GP_kernel_est = 1; % = 1: update hyper_parameters during execusion  
GP_kernel_est_timing = ... % = the timing of updating hyper_parameters during execusion: modify this to save computational time  
    [1,2,3,4,5,floor(logspace(1,5))]; 
%GP_kernel_est_timing = 1:1:nb_iter; % the setting used in the NIPS paper experiment to be fair with previous methods. But, this is not practical

% ------- parameters for gpml library (see the manual of gpml library for detail) -------
clear likfunc, clear meanfunc, clear covfunc, clear hyp.mean, clear hyp.cov
likfunc = @likGauss;  % likelihood function
meanfunc = @meanConst; %  mean function
covfunc = {@covMaterniso, 5}; % covariance function (kernel)
hyp.lik = -inf; hyp.mean = 0; ell = 1/4; sf = 1; hyp.cov = log([ell; sf]); % hyper-parameters

% ------- plot flag  ------- 
stop_per_iter = 0;  % stop per iteration

%% execute IMGPO
[x, fx, X_sample, F_sample, result] = IMGPO_stochastic_objective(f, d, x_input_domain, ...
          nb_iter, XI_max, ...
          GP_use, GP_kernel_est, GP_varsigma, likfunc, meanfunc, covfunc, hyp, ...
          result_diplay, result_save, GP_kernel_est_timing,...
          plot_func, plot_point, stop_per_iter);
