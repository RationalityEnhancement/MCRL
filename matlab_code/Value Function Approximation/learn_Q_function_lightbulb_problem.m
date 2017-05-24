%evaluate and tune Bayesian-SARSAQ with the features identified by linear
%regression
clear,close all,clc

addpath('../MatlabTools/') %change to your directory for MatlabTools
addpath('../metaMDP/')
addpath('../Supervised/')
addpath('../')


load ../../results/lightbulb_fit.mat

nr_actions=2;
nr_states=2;
gamma=1;

feature_names={'VPI','VOC_1','VOC_2','E[R|guess,b]','1'};
selected_features=[1;2;4];

nr_features=numel(selected_features);

costs=logspace(-3,-1,10);%0.01;%

for c=1:numel(costs)
    
    mdp=metaMDP(nr_actions,gamma,nr_features,costs(c));
    
    nr_episodes=5000;
    
    fexr=@(s,a,mdp) feature_extractor(s,a,mdp,selected_features);
    
    
    mdp.action_features=1:nr_features;
    
    sigma0=1;
    glm=BayesianGLM(nr_features,sigma0);
    glm.mu_0=[1;1;1];
    glm.mu_n=[1;1;1];
    [glm,avg_MSE,R_total]=BayesianSARSAQ(mdp,fexr,nr_episodes,glm);
    
    figure(),
    subplot(2,1,1)
    plot(smooth(avg_MSE,100))
    xlabel('Episode','FontSize',16)
    ylabel('Average MSE','FontSize',16)
    
    subplot(2,1,2)
    plot(smooth(R_total,100))
    xlabel('Episode','FontSize',16)
    ylabel('R_{total}','FontSize',16)
    
    
    w=glm.mu_n;
    figure()
    bar(w)
    ylabel('Learned Weights','FontSize',16)
    set(gca,'XTickLabel',feature_names(selected_features),'FontSize',16)
    
    %plot the corresponding fit to the Q-function
    nr_states=size(lightbulb_problem(c).mdp.states,1);
    
    for s=1:nr_states
        F(s,:)=fexr(lightbulb_problem(c).mdp.states(s,:),1,mdp);
    end
    
    valid_states=and(sum(lightbulb_problem(c).mdp.states,2)<=30,...
        sum(lightbulb_problem(c).mdp.states,2)>0);
    
    Q_hat(:,1)=F*w;
    Q_hat(:,2)=F(:,3);
    V_hat=max(Q_hat,[],2);
    
    R2(c)=corr(Q_hat(valid_states,1),lightbulb_problem(c).fit.Q_star(valid_states,1))
    
    lightbulb_problem(c).w_BSARSA=w;
    lightbulb_problem(c).Q_hat_BSARSA=Q_hat;
    lightbulb_problem(c).V_hat_BSARSA=V_hat;
    lightbulb_problem(c).R2_BSARSA=R2(c);
    
    fig_Q=figure()
    scatter(Q_hat(valid_states),lightbulb_problem(c).fit.Q_star(valid_states,1))
    set(gca,'FontSize',16)
    xlabel(['$\hat{Q}=',modelEquation(feature_names(selected_features),roundsd(w,4)),'$'],...
        'Interpreter','LaTeX','FontSize',16)
    ylabel('$Q^\star$','FontSize',16,'Interpreter','LaTeX')
    title(['Bayesian SARSA learns Q-function of 1-lightbulb meta-MDP, R^2=',num2str(roundsd(R2(c),4))],'FontSize',16)
    saveas(fig_Q,['../../results/figures/QFitToyProblemBayesianSARSA_c',int2str(c),'.fig'])
    saveas(fig_Q,['../../results/figures/QFitToyProblemBayesianSARSA_c',int2str(c),'.png'])
    
    %% Compute approximate PRs
    observe=1; guess=2;
    for s=1:nr_states-1
        approximate_PR(s,observe)=Q_hat(s,observe)-V_hat(s);
        approximate_PR(s,guess)=Q_hat(s,guess)-V_hat(s);
    end
    
    lightbulb_problem(c).approximate_PRs=approximate_PR;
end
save('../../results/lightbulb_fit_.mat','lightbulb_problem')