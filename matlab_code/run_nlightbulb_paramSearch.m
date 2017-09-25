

%% BSARSAQ

        load('../results/lightbulb_problem_opt.mat')
        load('../results/lightbulb_fit.mat')
        
alpha       = 0.1;   % learning rate
gamma       = 1;  % discount factor
epsilon     = 0.25;  % probability of a random action selection
nEpisodes = 1000;

        
nSims = 1;

addpath('MatlabTools/') %change to your directory for MatlabTools
addpath('metaMDP/')
addpath('Supervised/')
addpath('Value Function Approximation')

S = lightbulb_problem(1).mdp.states;
nr_actions=2;
nr_states=2;
gamma=1;

feature_names={'VPI','VOC_1','VOC_2','E[R|guess,b]','1'};
selected_features=[1;2;4];

nr_features=numel(selected_features);

costs=logspace(-3,-1/4,15);

mu1 = -10:2.5:10;
mu2 = -10:2.5:10;
mu3 = -10:2.5:10;
sigma0s = .01:.5:4;
R_BSARSAQ = nan(nSims,length(mu1),length(mu2),length(mu3),length(sigma0s),nEpisodes);

for m1 = 1:length(mu1)
    for m2 = 1:length(mu2)
        for m3 = 1:length(mu3)
            for s = 1:length(sigma0s)
                tic
mu = [mu1(m1);mu2(m2);mu3(m3)];
sigma0 = sigma0s(s);
cost = 0.001;
mdp=metaMDP(nr_actions,gamma,nr_features,cost);

nr_episodes=1000;

fexr=@(s,a,mdp) feature_extractor(s,a,mdp,selected_features);

mdp.action_features=1:nr_features;

glm=BayesianGLM(nr_features,sigma0);
glm.mu_0=mu;
glm.mu_n=mu;
GLM = glm; % for the parfor loop
% [glm,avg_MSE,R_total]=BayesianSARSAQ(mdp,fexr,nr_episodes,glm);
% R_optPR = nan(nSims,nEpisodes);
for i = 1:nSims
%     disp(num2str(i))
    [glm,avg_MSE,R_total]=BayesianSARSAQ(mdp,fexr,nr_episodes,GLM);
    R_BSARSAQ(i,m1,m2,m3,s,:) = R_total; %simulate_nlightbulb(nEpisodes,T,R,PRs_opt,epsilon,alpha,gamma);
end

toctmp
disp([m1,m2,m3,s])
            end
        end
    end
end