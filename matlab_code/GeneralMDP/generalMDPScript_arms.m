%% Set Variables
%  Unlike metaMDP we can only do one cost, nr_arms, depth at a time

setenv('DYLD_LIBRARY_PATH', '/usr/local/bin/');

addpath('./GeneralMDP/')
addpath('./MatlabTools')

rewardCorrect = 1;
constantArm = -1;
rewardIncorrect = 0;
discount = 1;

nrs_arms= 2:6;
nr_balls = 8;

% costs = logspace(-5,-1,10);
% method_ers = zeros(numel(costs),5);
% method_sterrs = zeros(numel(costs),5);

method_ers = zeros(numel(nrs_arms),5);
method_sterrs = zeros(numel(nrs_arms),5);

for c=1:numel(nrs_arms)
    nr_arms=nrs_arms(c);
    cost = 0.007;
tic 
 general_backwardsInduction %update for reward
toc

tic 
 general_regression %update for reward
toc

cd('./Value Function Approximation')
tic
    general_BSARSA %update for reward
toc
cd('../')

tic
 general_metaGreedy
toc

% tic
%  general_BO
% toc

gen_simulate

method_ers(c,:) = mean(samples);
method_sterrs(c,:) = std(samples);
end

gen_results.ers = method_ers;
gen_results.sterr = method_sterrs;

save('../results/gen_results.mat','gen_results') 