%% test semi-gradient SARSA that learns Q
addpath('~/Dropbox/PhD/MatlabTools/')

nr_arms=3;
gamma=1;

mdp=generalMDP(nr_arms,gamma);

epsilon=0.1;
nr_episodes=3000;
fexr=@(s,a,mdp) feature_extractor(s',a,mdp);

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
addpath('~/Dropbox/PhD/MatlabTools/')

nr_arms=3;
gamma=1;

mdp=generalMDP(nr_arms,gamma);

nr_episodes=1000;
fexr=@(s,a,mdp) feature_extractor(s,a,mdp);

nr_features=nr_arms*mdp.features_per_a+8; sigma0=1;
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