addpath('./MatlabTools')
addpath('./metaMDP/')
addpath('./Supervised/')
addpath('./Value Function Approximation')
addpath('./')

nTrials = 30;
rewardCorrect = 1;
rewardIncorrect = -1;
discount = 1;
costs=logspace(-3,-1/4,15);
plot_indicator = 0;

tic 
    metaMDP_backwardsInduction
toc 

tic
    metaMDP_regression
toc

cd('./Value Function Approximation')
tic
    metaMDP_BSARSA
toc
cd('../')

tic
metaMDP_metaGreedy
toc

tic
 metaMDP_BO
toc

simulate2