%solve Gaussian meta-level MDP with backwards induction

%problem='MouselabMDP';
problem='GaussianMetaMDP';

%1. Define meta-level MDP

if strcmp(problem,'MouselabMDP')
    start_state.delta_mu=0;
    start_state.sigma=1;
    cost=0.01;
    
    [T,R,states]=oneArmedMouselabMDP(nr_cells,mu_reward,sigma_reward,cost);
    meta_MDP=MouselabMDPMetaMDP(add_pseudorewards,pseudoreward_type,mean_payoff,std_payoff,object_level_MDPs);
    
elseif strcmp(problem,'GaussianMetaMDP')
    nr_cells=4;
    mu_reward=5;
    sigma_reward=10;
    cost=0.01;
    [T,R,states]=GaussianSamplingMetaMDP(start_state,cost);
    
    meta_MDP=GaussianMetaMDP(0,1);
end

nr_states=size(T,1);

horizon=10;
gamma=1;

[V, optimal_policy, ~] = mdp_finite_horizon(T, R, gamma, horizon);

%compute the VOC
for t=1:horizon-1
    for from=1:(nr_states-1)
        VOC(from,t)=dot(T(from,:,1),V(:,t+1))-R(from,end,2)-cost;
        VPIs(from)=valueOfPerfectInformation([states.MUs(from),0],[states.SIGMAs(from),0],1);
    end
    VOC(nr_states,t)=0;
end

%%
for t=1:4
    
    fig1=figure(1)
    subplot(4,1,t)
    imagesc(states.delta_mu_values,states.sigma_values,reshape(V(1:end-1,t),size(states.MUs)))
    xlabel('\mu','FontSize',16)
    ylabel('\sigma','FontSize',16)
    title(['Optimal Value Function, Step ',int2str(t)],'FontSize',18)
    colorbar()
    
    fig2=figure(2)
    subplot(4,1,t)
    imagesc(states.delta_mu_values,states.sigma_values,reshape(optimal_policy(1:end-1,t),size(states.MUs)))
    xlabel('\Delta\mu','FontSize',16)
    ylabel('\sigma','FontSize',16)
    title(['Optimal Policy, Step ',int2str(t)],'FontSize',18)
    
    fig3=figure(3)
    subplot(4,1,t)
    imagesc(states.delta_mu_values,states.sigma_values,reshape(VOC(1:end-1,t),size(states.MUs)))
    xlabel('\Delta\mu','FontSize',16)
    ylabel('\sigma','FontSize',16)
    title(['VOC(sample), Step ',int2str(t)],'FontSize',18)
    colorbar
end

[r,p]=corr(VOC(:,end),VOC(:,1))
X=[VOC(1:end-1,end),VPIs(:),ones(size(VOC(1:end-1,1)))];
[beta,beta_int,residuals,r_int,stats]=regress(VOC(1:end-1,1),[VOC(1:end-1,end),VPIs(:),ones(size(VOC(1:end-1,1)))]);
VOC_hat = X*beta;

[beta_restricted,beta_int_restricted,residuals_restricted,r_int_restricted,stats_restricted]=...
    regress(VOC(1:end-1,1),[VOC(1:end-1,end),ones(size(VOC(1:end-1,1)))]);

[r_VOC1_VOC,p]=corr(VOC(1:end-1,end),VOC(1:end-1,1))
[r_VPI_VOC,p]=corr(VPIs(:),VOC(1:end-1,1))
[r_VOC1_VPI,p]=corr(VOC(1:end-1,end),VPIs(:))

figure()
imagesc(states.delta_mu_values,states.sigma_values,reshape(residuals,size(states.MUs)))
xlabel('\mu','FontSize',16)
ylabel('\sigma','FontSize',16)
colorbar()

fig4=figure(4)
plot(VOC_hat,VOC(1:end-1,1),'*')
xlim([-0.2,0.25]),ylim([-0.2,0.25])
xlabel('Prediction','FontSize',16)
ylabel('VOC','FontSize',16)
title(['VOC=',num2str(beta(1)),'\times VOC_1 + ',num2str(beta(2)),'\times VPI + ',num2str(beta(3)),', R^2=',num2str(stats(1))],'FontSize',16)

fig5=figure(5)
imagesc(states.delta_mu_values,states.sigma_values,reshape(VOC(1:end-1,1)-VOC(1:end-1,end-1),size(states.MUs)))
xlabel('\Delta\mu','FontSize',16)
ylabel('\sigma','FontSize',16)
title('VOC-VOC_1','FontSize',18)
colorbar()

V_hat=max(0,VOC(:,1)+R(:,end,2)) %V*(b)=max_c {VOC(b,c)+E[R|act,b]}
figure()
plot(V_hat,V(:,1),'x')

figure()
imagesc(states.delta_mu_values,states.sigma_values,reshape(V_hat(1:end-1),size(states.MUs)))
colorbar()


delta_V=V(1:end-1,1)-V_hat(1:end-1);
figure()
imagesc(states.delta_mu_values,states.sigma_values,reshape(delta_V,size(states.MUs)))
colorbar()



%% plot VOC_n as a function of n for mu=0, sigma=1

sigma_plot=[1,0.8,0.6];
mu_plot=[0,0.1,0.2];
[mu_grid,sigma_grid]=meshgrid(mu_plot,sigma_plot);


i=0;
for m=1:numel(mu_plot)
    for s=1:numel(sigma_plot)
        i=i+1;
        [~,m_ind]=min(abs(states.delta_mu_values-mu_plot(m)));
        [~,s_ind]=min(abs(states.sigma_values-sigma_plot(s)));
        state_index=sub2ind(size(states.MUs),s_ind,m_ind);
        
        VPI=valueOfPerfectInformation([mu_plot(m),0],[sigma_plot(s),0],1);
        
        fig6=figure(6)
        subplot(3,3,i)
        plot(VOC(state_index,end-1:-1:1)),hold on,
        ylabel('VOC_n','FontSize',16)
        xlabel('n','FontSize',16)
        %plot(5,VPI,'rx')
        %legend('VOC_n','VPI')
        title(['\mu=',num2str(mu_plot(m)),', \sigma=',...
            num2str(sigma_plot(s)),', VPI=',num2str(roundsd(VPI,2))],'FontSize',16)
    end
end

%% evaluate the policy suggested by the weights
nr_regressors=3;
sigma=0.001;
glm=BayesianGLM(nr_regressors,sigma)
glm.mu_0=beta;
glm.mu_n=beta'
policy=@(state,mdp) contextualThompsonSampling(state,GaussianSamplingMetaMDP,glm(best_run))

[R_total,problems,states,chosen_actions,indices]=...
    inspectPolicyGeneral(GaussianSamplingMetaMDP,policy,nr_episodes)

%% try learning the policy

