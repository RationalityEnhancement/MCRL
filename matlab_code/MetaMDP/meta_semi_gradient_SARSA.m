%% This is the main function for training
% Test semi-gradient SARSA that learns Q
% Assumed that user is in matlab_code folder
% value_guess, policy_compare, and simulate are for metrics of evaluation
addpath('~/Documents/MATLAB/MatlabTools/') %change to your directory for MatlabTools
addpath('Value Function Approximation/')
addpath('metaMDP/')
addpath('Supervised/')

nr_actions=2;
nr_states=2;
gamma=1;

mdp=metaMDP(nr_actions,gamma);

epsilon=0.1;
nr_episodes=1000;
fexr=@(s,a,mdp) feature_extractor(s,a,mdp);

[w,avg_MSE]=semiGradientSARSAQ(mdp,fexr,nr_episodes,epsilon);
w1 = w;
figure(),
plot(avg_MSE)
xlabel('Episode','FontSize',16)
ylabel('Average MSE','FontSize',16)

figure()
bar(w)
ylabel('Learned Weights','FontSize',16)

%% Test Bayesian SARSA-Q
addpath('~/Documents/MATLAB/MatlabTools/') %change to your directory for MatlabTools
addpath('Value Function Approximation/')
addpath('metaMDP/')
addpath('Supervised/')

nr_actions=2;
nr_states=2;
gamma=1;

mdp=metaMDP(nr_actions,gamma);

nr_episodes=500;
fexr=@(s,a,mdp) feature_extractor(s,a,mdp);

nr_features=5; sigma0=1;
glm=BayesianGLM(nr_features,sigma0);
[glm,avg_MSE,R_total]=BayesianSARSAQ(mdp,fexr,nr_episodes,glm);

figure(),
plot(avg_MSE)
xlabel('Episode','FontSize',16)
ylabel('Average MSE','FontSize',16)

w=glm.mu_n;
w2=w;
figure()
bar(w)
ylabel('Learned Weights','FontSize',16)