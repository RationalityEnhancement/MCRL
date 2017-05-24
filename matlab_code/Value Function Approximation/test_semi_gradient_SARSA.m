%% test semi-gradient SARSA that learns Q
addpath('~/Dropbox/PhD/MatlabTools/')

nr_actions=2;
nr_states=2;
gamma=1;

mdp=metaMDP(nr_actions,gamma);

epsilon=0.1;
nr_episodes=10000;
% feature_extractor=@(s,a,mdp) [ones(1,size(s,2)); s+1; a];

[w,avg_MSE]=semiGradientSARSAQ(mdp,feature_extractor,nr_episodes,epsilon);

figure(),
plot(avg_MSE)
xlabel('Episode','FontSize',16)
ylabel('Average MSE','FontSize',16)

figure()
bar(w)
ylabel('Learned Weights','FontSize',16)

%% Test Bayesian SARSA-Q
addpath('~/Dropbox/PhD/MatlabTools/')

nr_actions=2;
nr_states=2;
gamma=1;

mdp=testMDP(nr_actions,gamma);

nr_episodes=100;
feature_extractor=@(s,a,mdp) [ones(1,size(s,2)); s+1; a];

nr_features=3; sigma0=1;
glm=BayesianGLM(nr_features,sigma0);
[glm,avg_MSE,R_total]=BayesianSARSAQ(mdp,feature_extractor,nr_episodes,glm);

figure(),
plot(avg_MSE)
xlabel('Episode','FontSize',16)
ylabel('Average MSE','FontSize',16)

w=glm.mu_n;
figure()
bar(w)
ylabel('Learned Weights','FontSize',16)