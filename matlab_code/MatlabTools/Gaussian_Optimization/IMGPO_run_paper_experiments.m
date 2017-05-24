addpath('/Users/Falk/Dropbox/PhD/MatlabTools/Gaussian_Optimization/test_function');
%% definition of the function to be optimized
sin1 = @(value) (sin(13 .* value) .* sin(27 .* value) ./ 2.0 + 0.5);
sin2D = @(value) (sin(13 .* value(:,1)) .* sin(27 .* value(:,1)) ./ 2.0 + 0.5) .* (sin(13 .* value(:,2)) .* sin(27 .* value(:,2)) ./ 2.0 + 0.5);
peak2D = @(value) 3*(1-value(:,2)).^2.*exp(-(value(:,2).^2) - (value(:,1)+1).^2) - 10*(value(:,2)/5 - value(:,2).^3 - value(:,1).^5).*exp(-value(:,2).^2-value(:,1).^2) - 1/3*exp(-(value(:,2)+1).^2 - value(:,1).^2);
Branin = @(value) - braninf(value);
Rosen2 = @(value) - ( ((1-value(:,1)).^2)+(100*((value(:,2)-(value(:,1).^2)).^2)) );
Hart3 = @(value) - hart3f(value);
Hart6 = @(value) - hart6f(value);
shekel5 = @(value) - shekelMf(value,5);

%% select which function from above we want to optimize
% d: dimension of the domain of f
% x_input_domain = input domain; 
%          e.g., = [-1 3; -3 3] means that domain(f) = {(x1,x2) : -1 <= x1 <= 3 and -3 <= x2 <= 3]} 
clear f;

%f = sin1; d=1; x_input_domain = [0 1]; 
f = sin2D; d=2; x_input_domain = [0 1; 0 1]; 
%f = peak2D; d=2; x_input_domain = [-3 3; -3 3]; 
%f = Branin; d=2; x_input_domain = [0 15; -5 10];             
%f = Rosen2; d=2; x_input_domain = [-5 10; -5 10; -5 10; -5 10; -5 10; -5 10; -5 10; -5 10; -5 10; -5 10];
%f = Hart3; d=3; x_input_domain = [0 1; 0 1; 0 1; 0 1; 0 1; 0 1]; 
%f = Hart6; d=6; x_input_domain = [0 1; 0 1; 0 1; 0 1; 0 1; 0 1]; 
%f = shekel5; d=4; x_input_domain = [0 10; 0 10; 0 10; 0 10]; 

%% illustration for a possibility of high-dim extention direction  
%A(:,1) = rand(1000,1);
%A(:,2) = ones(1000,1) - A(:,1);
%f = @(x) sin2D_high(x);  d=2; x_input_domain = [0 1; 0 1]; fmax = 0.975599143811574975870826165191829204559326171875^2; 
%f = @(x) f((A*x')');

%% parameter setting 
% ------- parameters for problem ------- 
nb_iter = 100;  % # of iteration

% ------- intermidiate results: display and save flag  ------- 
result_diplay = 1; % print intermidiate results
result_save = 1; % save intermidiate result and return to the third argument; x5 contains all the intermidiate results in [x1,x2,x3,x4,x5] = IMGPO()

% ------- plot flag  ------- 
plot_func = 1;  % plot objective function if the dimensionality is <= 2
plot_point = 1; % plot data points if the dimensionality is <= 2

%% execute IMGPO
[x, fx, X_sample, F_sample, result] = ...
                     IMGPO_default_run(f, d, x_input_domain, nb_iter, ...
                     result_diplay, result_save, plot_func, plot_point);
% main_IMGPO_default_run:
%   execute IMGPO with a default setting used in a NIPS paper
%   output: 
%      x = global optimizer 
%      fx = global optimal value f(x)
%   optinal output:
%      X_sample = sampled points 
%      F_sample = sampled values of f 
%      result = intermidiate results 
%               for each iteration t, result(t,:) = [N, n (split #), fmax_hat, rho_bar, xi_max, depth_T, time(s)] 
%   input: 
%      f = ovjective function (to be optimized)
%      d = input dimention of ovjective functio f
%      x_input_domain = input domain; 
%          e.g., = [-1 3; -3 3] means that domain(f) = {(x1,x2) : -1 <= x1 <= 3 and -3 <= x2 <= 3]} 
%      nb_iter = the number of iterations to perform
%   input display flag:
%      result_diplay = 1: print intermidiate results
%      result_save = 1: save intermidiate result and return as result
%      plot_func = 1: plot objective function if the dimensionality is <= 2
%      plot_point = 1; plot data points if the dimensionality is <= 2

%% output: display final result
fprintf(1,'Found: x = ');
fprintf(1,'%f, ', x);
fprintf(1,'\n f(x) = %f \n', fx);

%% output: write down the intermidiate result to a file 
fclose all;
fid_results = fopen('results.txt', 'w'); 
fprintf(fid_results, 'N, n (split #), fmax_hat, rho_bar, xi_max, depth_T, time(s), \n');
for i = 1 : size(result,1) 
  fprintf(fid_results, '%d, %d, %0.10f, %0.10f, %f, %d, %d, %0.10f \n', ...
      result(i,1),result(i,2),result(i,3),result(i,4),result(i,5),result(i,6),result(i,7));
end