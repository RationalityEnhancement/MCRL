%% illustration for a possibility of high-dim extention direction by a random projection
% a simplified version to illustrate the very simple idea 

addpath('test_function');
clear f;

for kkk = 1 : 30
LB_sv = -inf;
for jjj = 0 : 2    
A(:,1) = rand(1000,1);  % here, we use "rand" for simplicity only. 
A(:,2) = ones(1000,1) - A(:,1);

f = @(x) sin2D_high(x); d=2; x_input_domain = [0 1; 0 1]; fmax = 0.975599143811574975870826165191829204559326171875^2; 

f = @(x) f((A*x')');

%% parameter setting 
% ------- parameters for problem ------- 
nb_iter = 200;  % # of iteration

% ------- intermidiate results: display and save flag  ------- 
result_diplay = 1; % print intermidiate results
result_save = 1; % save intermidiate result and return to the third argument; x5 contains all the intermidiate results in [x1,x2,x3,x4,x5] = IMGPO()

% ------- plot flag  ------- 
plot_func = 0;  % plot objective function if the dimensionality is <= 2
plot_point = 0; % plot data points if the dimensionality is <= 2

%% execute IMGPO
[x, fx, X_sample, F_sample, result] = ...
                     IMGPO_default_run(f, d, x_input_domain, nb_iter, ...
                     result_diplay, result_save, plot_func, plot_point);
                 
result_sv(jjj*200+1:jjj*200+200,kkk) = max(result(1:200,3),LB_sv);
LB_dum = max(F_sample);
LB_sv = max(LB_sv,LB_dum);
end
end
result_sv2 = log10(fmax .* ones(600,30) - result_sv);
mean(result_sv2,2)
std(result_sv2')'


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