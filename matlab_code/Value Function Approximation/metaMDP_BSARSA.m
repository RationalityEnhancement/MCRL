%evaluate and tune Bayesian-SARSAQ with the features identified by linear
%regression
% clear,close all,clc

addpath('../MatlabTools/') %change to your directory for MatlabTools
addpath('../metaMDP/')
addpath('../Supervised/')
addpath('../')

S = lightbulb_problem(1).mdp.states;
nr_actions=2;
gamma=1;

feature_names={'VPI','VOI_1','VOC_2','E[R|guess,b]','cost','1'};
selected_features=[1;2;4;5];

nr_features=numel(selected_features);

costs=logspace(-3,-1/4,15);
mu = [0;0;1;0];
sigma0 = 0.2;


for c=1:numel(costs)
    mdp=metaMDP(nr_actions,gamma,nr_features,costs(c));
    
    nr_episodes=1000;
    
    fexr=@(s,a,mdp) feature_extractor(s,a,mdp,selected_features);  
    
    mdp.action_features=1:nr_features;
    
    glm=BayesianGLM(nr_features,sigma0);
    glm.mu_0=mu;
    glm.mu_n=mu;
    [glm,avg_MSE,R_total]=BayesianSARSAQ(mdp,fexr,nr_episodes,glm);
    
    w=glm.mu_n;
    
    %plot the corresponding fit to the Q-function
    for s=1:nr_states
        F(s,:)=fexr(lightbulb_problem(c).mdp.states(s,:),1,mdp);
    end
    
    valid_states=and(sum(lightbulb_problem(c).mdp.states,2)<=30,...
        sum(lightbulb_problem(c).mdp.states,2)>0);
    
    Q_hat(:,1)=F*w;
    Q_hat(:,2)=F(:,3);
    V_hat=max(Q_hat,[],2);
    
    qh = Q_hat(valid_states,1);
    qs = lightbulb_problem(c).fit.Q_star(valid_states,1);
    R2(c)=corr(Q_hat(valid_states,1),lightbulb_problem(c).fit.Q_star(valid_states,1));
    
    lightbulb_problem(c).BSARSA.w_BSARSA=w;
    lightbulb_problem(c).BSARSA.Q_hat_BSARSA=Q_hat;
    lightbulb_problem(c).BSARSA.V_hat_BSARSA=V_hat;
    lightbulb_problem(c).BSARSA.R2_BSARSA=R2(c);
    
    %% Compute approximate PRs
    observe=1; guess=2;
    for s=1:nr_states-1
        approximate_PR(s,observe)=Q_hat(s,observe)-V_hat(s);
        approximate_PR(s,guess)=Q_hat(s,guess)-V_hat(s);
    end
    
    lightbulb_problem(c).BSARSA.approximate_PRs=approximate_PR;

end
save('../../results/lightbulb_problem.mat','lightbulb_problem')