%% Do pseudo-rewards speed-up learning?
clear

alpha_clicks_per_cell=100;
beta_clicks_per_cell=100;
non_compensatoriness=1;
add_pseudoreward=true;

mdp_with_PR=MouselabMDP(non_compensatoriness,alpha_clicks_per_cell,beta_clicks_per_cell,true,'myopicVOC');
[~,mdp_with_PR]=mdp_with_PR.newEpisode();
mdp_without_PR=MouselabMDP(non_compensatoriness,alpha_clicks_per_cell,beta_clicks_per_cell,false);
[~,mdp_without_PR]=mdp_without_PR.newEpisode();

feature_extractor=@(s,a,mdp) mdp.extractStateActionFeatures(s,a);

epsilon=0.05;
nr_episodes=200;

parfor rep=1:50
    mdp=mdp_with_PR;
    [w_with_PR(:,rep),MSE_with_PR(:,rep),returns_with_PR(:,rep)]=...
        semiGradientSARSAQ(mdp,feature_extractor,nr_episodes,epsilon);
    
    mdp=mdp_without_PR;
    [w_without_PR(:,rep),MSE_without_PR(:,rep),returns_without_PR(:,rep)]=...
        semiGradientSARSAQ(mdp,feature_extractor,nr_episodes,epsilon);
end

avg_MSE_with_PR=mean(MSE_with_PR,2);
avg_MSE_without_PR=mean(MSE_without_PR,2);

R_total_with_PR=mean(returns_with_PR,2);
R_total_without_PR=mean(returns_without_PR,2);

bin_width=10;
episode_nrs=bin_width:bin_width:nr_episodes;
[avg_RMSE_PR,sem_RMSE_PR]=binnedAverage(sqrt(avg_MSE_with_PR),bin_width);
[avg_RMSE_noPR,sem_RMSE_noPR]=binnedAverage(sqrt(avg_MSE_without_PR),bin_width);
[avg_R_PR,sem_R_PR]=binnedAverage(R_total_with_PR,bin_width);
[avg_R_noPR,sem_R_noPR]=binnedAverage(R_total_without_PR,bin_width);

figure()
subplot(2,1,1)
errorbar(episode_nrs,avg_R_PR,sem_R_PR,'g-o','LineWidth',2), hold on
errorbar(episode_nrs,avg_R_noPR,sem_R_noPR,'r-o','LineWidth',2),
set(gca,'FontSize',16)
xlabel('Episode','FontSize',16)
ylabel('R_{total}','FontSize',16),
legend('with PR','without PR')
title('Semi-gradient SARSA (Q) in Mouselab Task','FontSize',18)
ylim([-3,25])
xlim([0,205])
%hold on
%plot(smooth(R_total,100),'r-')
%legend('RMSE','R_{total}')
xlabel('#Episodes','FontSize',16)
subplot(2,1,2)
errorbar(episode_nrs,avg_RMSE_PR,sem_RMSE_PR,'g-o','LineWidth',2), hold on
errorbar(episode_nrs,avg_RMSE_noPR,sem_RMSE_noPR,'r-o','LineWidth',2),
xlim([0,205])
set(gca,'FontSize',16)
xlabel('Episode','FontSize',16)
ylabel('RMSE','FontSize',16),
legend('with PR','without PR')
%hold on
%plot(smooth(R_total,100),'r-')
%legend('RMSE','R_{total}')
xlabel('#Episodes','FontSize',16)

feature_names={'mu(a*)','sigma(a*)','sigma(max EV)','remaining #clicks','mu(b)',...
    'sigma(b)','Expected Regret','E[max mu]','EV','myopic VOC', ...
    'sigma(g)', 'p(o)','max mu - mu(g)','early decision','decision on last click'};

figure()
subplot(2,1,1)
bar(mean(w_with_PR,2)),
ylim([0,0.3])
set(gca,'XTick',1:numel(feature_names),'XTickLabel',feature_names)
set(gca,'XTickLabelRotation',45,'FontSize',16)
ylabel('Learned Weights','FontSize',16)
title('With PR','FontSize',18)

subplot(2,1,2)
bar(mean(w_without_PR,2)),
ylim([0,0.3])
set(gca,'XTick',1:numel(feature_names),'XTickLabel',feature_names)
set(gca,'XTickLabelRotation',45,'FontSize',16)
ylabel('Learned Weights','FontSize',16)
title('Without PR','FontSize',18)

%estimate return of myopic meta-level policy
feature_extractor=@(s,mdp) mdp.extractStateFeatures(s);
[~,~,returns]=LinearTD0(mdp_without_PR,feature_extractor,1000)
avg_return_metalevel_greedy=mean(returns)

save PseudoRewardsAccelerateSARSAQ

%% Repeat the analysis above for the algorithm learning an approximation to the state-value function V
clear

alpha_clicks_per_cell=100;
beta_clicks_per_cell=100;
non_compensatoriness=1;
add_pseudoreward=true;

mdp_with_PR=MouselabMDP(non_compensatoriness,alpha_clicks_per_cell,beta_clicks_per_cell,true,'myopicVOC');
[~,mdp_with_PR]=mdp_with_PR.newEpisode();
mdp_without_PR=MouselabMDP(non_compensatoriness,alpha_clicks_per_cell,beta_clicks_per_cell,false);
[~,mdp_without_PR]=mdp_without_PR.newEpisode();

feature_extractor=@(s,mdp) mdp.extractStateFeatures(s);

epsilon=0.05;
nr_episodes=200;

parfor rep=1:20
    mdp=mdp_with_PR;
    [w_with_PR(:,rep),MSE_with_PR(:,rep),returns_with_PR(:,rep)]=...
        semiGradientSARSA(mdp,feature_extractor,nr_episodes,epsilon);
    
    mdp=mdp_without_PR;
    [w_without_PR(:,rep),MSE_without_PR(:,rep),returns_without_PR(:,rep)]=...
        semiGradientSARSA(mdp,feature_extractor,nr_episodes,epsilon);
end

avg_MSE_with_PR=mean(MSE_with_PR,2);
avg_MSE_without_PR=mean(MSE_without_PR,2);

R_total_with_PR=mean(returns_with_PR,2);
R_total_without_PR=mean(returns_without_PR,2);

figure()
subplot(2,1,1)
plot(smooth(sqrt(avg_MSE_with_PR),nr_episodes/10),'b-','LineWidth',2), hold on
plot(smooth(sqrt(avg_MSE_without_PR),nr_episodes/10),'r-','LineWidth',2),
set(gca,'FontSize',16)
xlabel('Episode','FontSize',16)
ylabel('RMSE','FontSize',16),
legend('with PR','without PR')
title('Semi-gradient SRS (V) in Mouselab Task','FontSize',18)
%hold on
%plot(smooth(R_total,100),'r-')
%legend('RMSE','R_{total}')
xlabel('#Episodes','FontSize',16)
subplot(2,1,2)
plot(smooth(R_total_with_PR,nr_episodes/10),'b-','LineWidth',2), hold on
plot(smooth(R_total_without_PR,nr_episodes/10),'r-','LineWidth',2),
set(gca,'FontSize',16)
xlabel('Episode','FontSize',16)
ylabel('R_{total}','FontSize',16),
legend('with PR','without PR')
%hold on
%plot(smooth(R_total,100),'r-')
%legend('RMSE','R_{total}')
xlabel('#Episodes','FontSize',16)

figure()
subplot(2,1,1)
bar(mean(w_without_PR,2)),set(gca,'XTickLabel',{'mu(a*)','sigma(a*)','sigma(max EV)','remaining #clicks','mu(b)',...
    'sigma(b)','Expected Regret','E[max mu]','EV','myopic VOC', ...
    'sigma(g)', 'p(o)','max mu - mu(g)'})
set(gca,'XTickLabelRotation',45,'FontSize',16)
ylabel('Learned Weights','FontSize',16)
title('Without PR','FontSize',18)

subplot(2,1,2)
bar(mean(w_with_PR,2)),set(gca,'XTickLabel',{'mu(a*)','sigma(a*)','sigma(max EV)','remaining #clicks','mu(b)',...
    'sigma(b)','Expected Regret','E[max mu]','EV','myopic VOC', ...
    'sigma(g)', 'p(o)','max mu - mu(g)'})
set(gca,'XTickLabelRotation',45,'FontSize',16)
ylabel('Learned Weights','FontSize',16)
title('With PR','FontSize',18)

save PseudorewardsAccelerateSARSAV

%% Repeat the experiments with pseudo-rewards based on the expected regret
clear

alpha_clicks_per_cell=100;
beta_clicks_per_cell=100;
non_compensatoriness=1;
add_pseudoreward=true;

mdp_with_PR=MouselabMDP(non_compensatoriness,alpha_clicks_per_cell,beta_clicks_per_cell,true,'regretReduction');
[~,mdp_with_PR]=mdp_with_PR.newEpisode();
mdp_without_PR=MouselabMDP(non_compensatoriness,alpha_clicks_per_cell,beta_clicks_per_cell,false);
[~,mdp_without_PR]=mdp_without_PR.newEpisode();

feature_extractor=@(s,a,mdp) mdp.extractStateActionFeatures(s,a);

epsilon=0.05;
nr_episodes=200;

parfor rep=1:50
    %for rep=1:1
    mdp=mdp_with_PR;
    [w_with_PR(:,rep),MSE_with_PR(:,rep),returns_with_PR(:,rep)]=...
        semiGradientSARSAQ(mdp,feature_extractor,nr_episodes,epsilon);
    
    mdp=mdp_without_PR;
    [w_without_PR(:,rep),MSE_without_PR(:,rep),returns_without_PR(:,rep)]=...
        semiGradientSARSAQ(mdp,feature_extractor,nr_episodes,epsilon);
end

avg_MSE_with_PR=mean(MSE_with_PR,2);
avg_MSE_without_PR=mean(MSE_without_PR,2);

R_total_with_PR=mean(returns_with_PR,2);
R_total_without_PR=mean(returns_without_PR,2);

bin_width=10;
episode_nrs=bin_width:bin_width:nr_episodes;
[avg_RMSE_PR,sem_RMSE_PR]=binnedAverage(sqrt(avg_MSE_with_PR),bin_width);
[avg_RMSE_noPR,sem_RMSE_noPR]=binnedAverage(sqrt(avg_MSE_without_PR),bin_width);
[avg_R_PR,sem_R_PR]=binnedAverage(R_total_with_PR,bin_width);
[avg_R_noPR,sem_R_noPR]=binnedAverage(R_total_without_PR,bin_width);

figure()
subplot(2,1,1)
errorbar(episode_nrs,avg_R_PR,sem_R_PR,'g-o','LineWidth',2), hold on
errorbar(episode_nrs,avg_R_noPR,sem_R_noPR,'r-o','LineWidth',2),
set(gca,'FontSize',16)
xlabel('Episode','FontSize',16)
ylabel('R_{total}','FontSize',16),
legend('with PR','without PR')
title('Semi-gradient SARSA (Q) in Mouselab Task','FontSize',18)
ylim([-3,25])
xlim([0,205])
%hold on
%plot(smooth(R_total,100),'r-')
%legend('RMSE','R_{total}')
xlabel('#Episodes','FontSize',16)
subplot(2,1,2)
errorbar(episode_nrs,avg_RMSE_PR,sem_RMSE_PR,'g-o','LineWidth',2), hold on
errorbar(episode_nrs,avg_RMSE_noPR,sem_RMSE_noPR,'r-o','LineWidth',2),
xlim([0,205])
set(gca,'FontSize',16)
xlabel('Episode','FontSize',16)
ylabel('RMSE','FontSize',16),
legend('with PR','without PR')
%hold on
%plot(smooth(R_total,100),'r-')
%legend('RMSE','R_{total}')
xlabel('#Episodes','FontSize',16)

feature_names={'mu(a*)','sigma(a*)','sigma(max EV)','remaining #clicks','mu(b)',...
    'sigma(b)','Expected Regret','E[max mu]','EV','myopic VOC', ...
    'sigma(g)', 'p(o)','max mu - mu(g)','early decision','decision on last click'};

figure()
subplot(2,1,1)
bar(mean(w_with_PR,2)),
ylim([0,0.2])
set(gca,'XTick',1:numel(feature_names),'XTickLabel',feature_names)
set(gca,'XTickLabelRotation',45,'FontSize',16)
ylabel('Learned Weights','FontSize',16)
title('With PR','FontSize',18)

subplot(2,1,2)
bar(mean(w_without_PR,2)),
ylim([0,0.2])
set(gca,'XTick',1:numel(feature_names),'XTickLabel',feature_names)
set(gca,'XTickLabelRotation',45,'FontSize',16)
ylabel('Learned Weights','FontSize',16)
title('Without PR','FontSize',18)

%estimate return of myopic meta-level policy
feature_extractor=@(s,mdp) mdp.extractStateFeatures(s);
[~,~,returns]=LinearTD0(mdp_without_PR,feature_extractor,1000)
avg_return_metalevel_greedy=mean(returns)

save regretReductionSARSAQ

%% simulate object-level RL in the pilot task

N=6;
horizon=inf;
gamma=1-10*eps;
p_end=1/N;

transition_matrices(:,:,1) = [0,1-p_end,0,0,0,0, p_end; 0,0,1-p_end,0,0,0,p_end; 0,0,0,1-p_end,0,0,p_end; 0,0,0,0,1-p_end,0,p_end; 0,0,0,0,0,1-p_end,p_end; 1-p_end,0,0,0,0,0,p_end; 0,0,0,0,0,0,p_end];
transition_matrices(:,:,2) = [0,0,0,1-p_end,0,0,p_end; 0,0,0,0,1-p_end,0,p_end; 0,0,0,0,0,1-p_end,p_end; 0,1-p_end,0,0,0,0,p_end; 1-p_end,0,0,0,0,0,p_end; 0,0,1-p_end,0,0,0,p_end; 0 0 0 0 0 0 p_end];

nr_actions=size(transition_matrices,3);

%Rewards
small_positive=3;
small_negative=-3;
large_positive=14;
large_negative=-7;

rewards(1,:,1) = [0,large_positive,0,0,0,0,large_positive];
rewards(2,:,1) = [0,0,small_negative,0,0,0,small_negative];
rewards(3,:,1) = [0,0,0,small_negative,0,0,small_negative];
rewards(4,:,1) = [0,0,0,0,small_negative,0,small_negative];
rewards(5,:,1) = [0,0,0,0,0,small_negative,small_negative];
rewards(6,:,1) = [small_negative,0,0,0,0,0,small_negative];
rewards(7,:,1) = [0,0,0,0,0,0,0];

rewards(1,:,2) = [0,0,0,small_positive,0,0,small_positive];
rewards(2,:,2) = [0,0,0,0,large_negative,0,large_negative];
rewards(3,:,2) = [0,0,0,0,0,large_negative,large_negative];
rewards(4,:,2) = [0,small_positive,0,0,0,0,small_positive];
rewards(5,:,2) = [large_negative,0,0,0,0,0,large_negative];
rewards(6,:,2) = [0,0,small_positive,0,0,0,small_positive];
rewards(7,:,2) = [0,0,0,0,0,0,0];

nr_states=7;
states=1:nr_states;
actions=1:2;

P0=1/(nr_states-1)*[ones(nr_states-1,1);0]; %uniform over all states, except for the terminal state
terminal_states=[7];

mdp=MDPSimulator(states,actions,transition_matrices,rewards,P0,terminal_states);

optimal_pseudorewards=optimalPseudoRewards(transition_matrices,rewards,horizon,gamma,false);
optimal_pseudorewards(:,:,2)=optimal_pseudorewards(:,:,1);
optimal_pseudorewards(7,:,:)=0;
for from=1:6
    next_state1=find(transition_matrices(from,1:6,1)>0);
    next_state2=find(transition_matrices(from,1:6,1)>0);
    optimal_pseudorewards(from,7,1)=optimal_pseudorewards(from,next_state1,1);
    optimal_pseudorewards(from,7,2)=optimal_pseudorewards(from,next_state2,1);
end

mdp_with_PR=MDPSimulator(states,actions,transition_matrices,rewards,P0,terminal_states,optimal_pseudorewards);

q_learning_agent=ModelFreeSystem(mdp);

nr_trials=1000;
nr_iterations=100*nr_trials;

nr_repetitions=1000;
parfor r=1:nr_repetitions
    [agents_without_PR(r),R_total_without_PR(:,r),episodes_without_PR(:,r),avg_discrepancy_without_PR(:,r)]=...
        mdp.simulate(q_learning_agent,nr_trials,1)
    
    [agents_with_PR(r),R_total_with_PR(:,r),episodes_with_PR(:,r),avg_discrepancy_with_PR(:,r)]=...
        mdp_with_PR.simulate(q_learning_agent,nr_trials,1)
    
    
    %[Q, V, policy, mean_discrepancy_without_PR(:,r),total_reward_without_PR(:,r)] = mdp_Q_learning2(transition_matrices, rewards, gamma, nr_iterations,25);
    %[Q_with_PR, V_with_PR, policy_with_PR(:,r), mean_discrepancy_with_PR(:,r),total_reward_with_PR(:,r)] = mdp_Q_learning2(transition_matrices, rewards+optimal_pseudorewards, gamma, nr_iterations,25);
end

for sim=1:nr_repetitions
    returns_without_PR(:,sim)=R_total_without_PR(sim).returns;
    returns_with_PR(:,sim)=R_total_with_PR(sim).returns;
end

figure()
errorbar(1:nr_trials,mean(returns_without_PR,2),sem(returns_without_PR(1:nr_trials,:),2),'LineWidth',2),hold on,
xlim([-10,nr_trials+10])
errorbar(1:nr_trials,mean(returns_with_PR,2),sem(returns_without_PR,2),'LineWidth',2)
xlabel('Nr. Trials (Episodes)','FontSize',18)
ylabel('Total reward','FontSize',18)
set(gca,'FontSize',16)
legend('no PR','with PR')

shown_nr_trials=24;
figure()
errorbar(1:shown_nr_trials,mean(mean_discrepancy_without_PR(1:shown_nr_trials,:),2),sem(mean_discrepancy_without_PR(1:shown_nr_trials,:),2),'LineWidth',2),hold on,
errorbar(1:shown_nr_trials,mean(mean_discrepancy_with_PR(1:shown_nr_trials,:),2),sem(mean_discrepancy_with_PR(1:shown_nr_trials,:),2),'LineWidth',2)
xlabel('Nr. Trials (Episodes)','FontSize',18)
ylabel('Mean Discrepancy','FontSize',18)
set(gca,'FontSize',16)
legend('no PR','with PR')


%% Simulate metacognitive RL
N=6;
horizon=inf;
gamma=1-10*eps;
p_end=1/N;

transition_matrices(:,:,1) = [0,1-p_end,0,0,0,0, p_end; 0,0,1-p_end,0,0,0,p_end; 0,0,0,1-p_end,0,0,p_end; 0,0,0,0,1-p_end,0,p_end; 0,0,0,0,0,1-p_end,p_end; 1-p_end,0,0,0,0,0,p_end; 0,0,0,0,0,0,p_end];
transition_matrices(:,:,2) = [0,0,0,1-p_end,0,0,p_end; 0,0,0,0,1-p_end,0,p_end; 0,0,0,0,0,1-p_end,p_end; 0,1-p_end,0,0,0,0,p_end; 1-p_end,0,0,0,0,0,p_end; 0,0,1-p_end,0,0,0,p_end; 0 0 0 0 0 0 p_end];

nr_actions=size(transition_matrices,3);

%Rewards
small_positive=3;
small_negative=-3;
large_positive=14;
large_negative=-7;

rewards(1,:,1) = [0,large_positive,0,0,0,0,large_positive];
rewards(2,:,1) = [0,0,small_negative,0,0,0,small_negative];
rewards(3,:,1) = [0,0,0,small_negative,0,0,small_negative];
rewards(4,:,1) = [0,0,0,0,small_negative,0,small_negative];
rewards(5,:,1) = [0,0,0,0,0,small_negative,small_negative];
rewards(6,:,1) = [small_negative,0,0,0,0,0,small_negative];
rewards(7,:,1) = [0,0,0,0,0,0,0];

rewards(1,:,2) = [0,0,0,small_positive,0,0,small_positive];
rewards(2,:,2) = [0,0,0,0,large_negative,0,large_negative];
rewards(3,:,2) = [0,0,0,0,0,large_negative,large_negative];
rewards(4,:,2) = [0,small_positive,0,0,0,0,small_positive];
rewards(5,:,2) = [large_negative,0,0,0,0,0,large_negative];
rewards(6,:,2) = [0,0,small_positive,0,0,0,small_positive];
rewards(7,:,2) = [0,0,0,0,0,0,0];

states=1:7;
actions=1:2;
terminal_states=[7];
P0=[1/6*ones(6,1);0];

%simulate performance of planners with different horizons
nr_episodes=100;
nr_simulations=100;
for h=1:numel(horizons)
    planner=PlanningAgent([horizons(h)],mdp);
    
    [fixed_horizon_planners{h},R_total_fixed_horizon{h},episodes_fixed_horizon{h}]=...
        mdp.simulate(planner,nr_episodes,nr_simulations)
    
    avg_total_return(h)=mean(R_total_fixed_horizon{h}.mean)
    sem_total_return(h)=mean(R_total_fixed_horizon{h}.sem)/sqrt(nr_episodes)
end

figure()
barwitherr(sem_total_return,avg_total_return)
set(gca,'XTickLabel',num2str(horizons'),'FontSize',16)
xlabel('Planning Horizon','FontSize',18)
ylabel('Avg. Return','FontSize',18)



mdp=MDPSimulator(states,actions,transition_matrices,rewards,P0,terminal_states);
horizons=[1,10];
planner=PlanningAgent(horizons,mdp);

%nr_episodes=1000;
%nr_simulations=1000;
nr_episodes=400;
nr_simulations=1000;
[agent_without_PR,R_total_without_PR,episodes_without_PR]=mdp.simulate(planner,nr_episodes,nr_simulations)

pseudorewards=optimalPseudoRewards(transition_matrices,rewards,horizon,gamma,false);
pseudorewards(:,end)=0; pseudorewards(end,:)=0;
mdp_with_PR=MDPSimulator(states,actions,transition_matrices,rewards,P0,terminal_states,pseudorewards);
planner=PlanningAgent(horizons,mdp_with_PR);
[agent_with_PR,R_total_with_PR,episodes_with_PR]=mdp_with_PR.simulate(planner,nr_episodes,nr_simulations)

bin_width=10;
[avg_returns_withoutPR,sem_returns_withoutPR]=binnedAverage(R_total_without_PR.mean(:),bin_width)
[avg_returns_withPR,sem_returns_withPR]=binnedAverage(R_total_with_PR.mean(:),bin_width)

nr_bins=numel(avg_returns_withoutPR);
bins=bin_width:bin_width:(bin_width*nr_bins);
fid_LC=figure()
errorbar(bins,avg_returns_withoutPR,sem_returns_withoutPR),hold on
errorbar(bins,avg_returns_withPR,sem_returns_withPR),hold on
xlim([-10,nr_episodes+10])
ylabel('Avg. Return','FontSize',18)
xlabel('Trial Number','FontSize',18)
set(gca,'FontSize',16)
legend('without PR','with PR')
saveas(fid_LC,'MetalevelLearningCurves.fig')

for sim=1:nr_simulations
    w_with_PR(:,sim)=agent_with_PR{sim}.w_Q_meta;
    w_without_PR(:,sim)=agent_without_PR{sim}.w_Q_meta;
end

fid_Q=figure()
subplot(2,1,1)
imagesc(reshape(mean(w_with_PR,2),[2,7])),colorbar()
set(gca,'YTick',1:2,'YTickLabel',{'Fast (h=1)','Slow (h=10)'})
set(gca,'XTickLabel',{'Smiths-','Williams-','Jones-','Browns-','Clarks-','Bakers-'})
title('Meta-level Q-function learned with PRs','FontSize',18)
xlabel('State','FontSize',16)
ylabel('Planning System','FontSize',16)
subplot(2,1,2)
imagesc(reshape(mean(w_without_PR,2),[2,7])),colorbar()
set(gca,'YTick',1:2,'YTickLabel',{'Fast (h=1)','Slow (h=10)'})
set(gca,'XTickLabel',{'Smiths-','Williams-','Jones-','Browns-','Clarks-','Bakers-'})
title('Meta-level Q-function learned without PRs','FontSize',18)
xlabel('State','FontSize',16)
ylabel('Planning System','FontSize',16)
saveas(fid_Q,'MetalevelQFunctions.fig')

%% Integrating object-level and meta-level RL
N=6;
horizon=inf;
gamma=1-10*eps;
p_end=1/N;

transition_matrices(:,:,1) = [0,1-p_end,0,0,0,0, p_end; 0,0,1-p_end,0,0,0,p_end; 0,0,0,1-p_end,0,0,p_end; 0,0,0,0,1-p_end,0,p_end; 0,0,0,0,0,1-p_end,p_end; 1-p_end,0,0,0,0,0,p_end; 0,0,0,0,0,0,p_end];
transition_matrices(:,:,2) = [0,0,0,1-p_end,0,0,p_end; 0,0,0,0,1-p_end,0,p_end; 0,0,0,0,0,1-p_end,p_end; 0,1-p_end,0,0,0,0,p_end; 1-p_end,0,0,0,0,0,p_end; 0,0,1-p_end,0,0,0,p_end; 0 0 0 0 0 0 p_end];

nr_actions=size(transition_matrices,3);

%Rewards
small_positive=3;
small_negative=-3;
large_positive=14;
large_negative=-7;

rewards(1,:,1) = [0,large_positive,0,0,0,0,large_positive];
rewards(2,:,1) = [0,0,small_negative,0,0,0,small_negative];
rewards(3,:,1) = [0,0,0,small_negative,0,0,small_negative];
rewards(4,:,1) = [0,0,0,0,small_negative,0,small_negative];
rewards(5,:,1) = [0,0,0,0,0,small_negative,small_negative];
rewards(6,:,1) = [small_negative,0,0,0,0,0,small_negative];
rewards(7,:,1) = [0,0,0,0,0,0,0];

rewards(1,:,2) = [0,0,0,small_positive,0,0,small_positive];
rewards(2,:,2) = [0,0,0,0,large_negative,0,large_negative];
rewards(3,:,2) = [0,0,0,0,0,large_negative,large_negative];
rewards(4,:,2) = [0,small_positive,0,0,0,0,small_positive];
rewards(5,:,2) = [large_negative,0,0,0,0,0,large_negative];
rewards(6,:,2) = [0,0,small_positive,0,0,0,small_positive];
rewards(7,:,2) = [0,0,0,0,0,0,0];

states=1:7;
actions=1:2;
terminal_states=[7];
P0=[1/6*ones(6,1);0];

mdp=MDPSimulator(states,actions,transition_matrices,rewards,P0,terminal_states);
horizons=[0,10];
agent=DualSystemsAgent(horizons,mdp);

%nr_episodes=1000;
%nr_simulations=1000;
nr_episodes=1000;
nr_simulations=1000;
[agent_without_PR,R_total_without_PR,episodes_without_PR]=mdp.simulate(agent,nr_episodes,nr_simulations)

pseudorewards=optimalPseudoRewards(transition_matrices,rewards,horizon,gamma,false);
pseudorewards(:,end)=0; pseudorewards(end,:)=0;
mdp_with_PR=MDPSimulator(states,actions,transition_matrices,rewards,P0,terminal_states,pseudorewards);
agent=DualSystemsAgent(horizons,mdp);
[agent_with_PR,R_total_with_PR,episodes_with_PR]=mdp_with_PR.simulate(agent,nr_episodes,nr_simulations)

bin_width=10;
[avg_returns_withoutPR,sem_returns_withoutPR]=binnedAverage(R_total_without_PR.mean(:),bin_width)
[avg_returns_withPR,sem_returns_withPR]=binnedAverage(R_total_with_PR.mean(:),bin_width)

nr_bins=numel(avg_returns_withoutPR);
bins=bin_width:bin_width:(bin_width*nr_bins);
fid_LC=figure()
errorbar(bins,avg_returns_withoutPR,sem_returns_withoutPR),hold on
errorbar(bins,avg_returns_withPR,sem_returns_withPR),hold on
xlim([-5,nr_episodes+5])
ylabel('Avg. Return','FontSize',18)
xlabel('Trial Number','FontSize',18)
set(gca,'FontSize',16)
legend('without PR','with PR')
saveas(fid_LC,'MetalevelLearningCurves.fig')

for sim=1:nr_simulations
    w_with_PR(:,sim)=agent_with_PR{sim}.w_Q_meta;
    w_without_PR(:,sim)=agent_without_PR{sim}.w_Q_meta;
end

fid_Q=figure()
subplot(2,1,1)
imagesc(reshape(mean(w_with_PR,2),[2,7])),colorbar()
set(gca,'YTick',1:2,'YTickLabel',{'Habits','Planning'})
set(gca,'XTickLabel',{'Smiths-','Williams-','Jones-','Browns-','Clarks-','Bakers-','Terminal State'})
title('Meta-level Q-function learned with PRs','FontSize',18)
xlabel('State','FontSize',16)
ylabel('Planning System','FontSize',16)
subplot(2,1,2)
imagesc(reshape(mean(w_without_PR,2),[2,7])),colorbar()
set(gca,'YTick',1:2,'YTickLabel',{'Habits','Planning'})
set(gca,'XTickLabel',{'Smiths-','Williams-','Jones-','Browns-','Clarks-','Bakers-','Terminal State'})
title('Meta-level Q-function learned without PRs','FontSize',18)
xlabel('State','FontSize',16)
ylabel('Planning System','FontSize',16)
saveas(fid_Q,'MetalevelQFunctionsIntegrated1000Trials.fig')

%% evaluate sensitivity to the cost of planning

N=6;
horizon=inf;
gamma=1-10*eps;
p_end=1/N;

transition_matrices(:,:,1) = [0,1-p_end,0,0,0,0, p_end; 0,0,1-p_end,0,0,0,p_end; 0,0,0,1-p_end,0,0,p_end; 0,0,0,0,1-p_end,0,p_end; 0,0,0,0,0,1-p_end,p_end; 1-p_end,0,0,0,0,0,p_end; 0,0,0,0,0,0,p_end];
transition_matrices(:,:,2) = [0,0,0,1-p_end,0,0,p_end; 0,0,0,0,1-p_end,0,p_end; 0,0,0,0,0,1-p_end,p_end; 0,1-p_end,0,0,0,0,p_end; 1-p_end,0,0,0,0,0,p_end; 0,0,1-p_end,0,0,0,p_end; 0 0 0 0 0 0 p_end];

nr_actions=size(transition_matrices,3);

%Rewards
small_positive=3;
small_negative=-3;
large_positive=14;
large_negative=-7;

rewards(1,:,1) = [0,large_positive,0,0,0,0,large_positive];
rewards(2,:,1) = [0,0,small_negative,0,0,0,small_negative];
rewards(3,:,1) = [0,0,0,small_negative,0,0,small_negative];
rewards(4,:,1) = [0,0,0,0,small_negative,0,small_negative];
rewards(5,:,1) = [0,0,0,0,0,small_negative,small_negative];
rewards(6,:,1) = [small_negative,0,0,0,0,0,small_negative];
rewards(7,:,1) = [0,0,0,0,0,0,0];

rewards(1,:,2) = [0,0,0,small_positive,0,0,small_positive];
rewards(2,:,2) = [0,0,0,0,large_negative,0,large_negative];
rewards(3,:,2) = [0,0,0,0,0,large_negative,large_negative];
rewards(4,:,2) = [0,small_positive,0,0,0,0,small_positive];
rewards(5,:,2) = [large_negative,0,0,0,0,0,large_negative];
rewards(6,:,2) = [0,0,small_positive,0,0,0,small_positive];
rewards(7,:,2) = [0,0,0,0,0,0,0];

states=1:7;
actions=1:2;
terminal_states=[7];
P0=[1/6*ones(6,1);0];

mdp=MDPSimulator(states,actions,transition_matrices,rewards,P0,terminal_states);
horizons=[0,10];
agent=DualSystemsAgent(horizons,mdp);

%nr_episodes=1000;
%nr_simulations=1000;
nr_episodes=1000;
nr_simulations=1000;
agent.system2.cost_of_planning=0;
[agent_without_PR,R_total_without_PR,episodes_without_PR]=mdp.simulate(agent,nr_episodes,nr_simulations)

pseudorewards=optimalPseudoRewards(transition_matrices,rewards,horizon,gamma,false);
pseudorewards(:,end)=0; pseudorewards(end,:)=0;
mdp_with_PR=MDPSimulator(states,actions,transition_matrices,rewards,P0,terminal_states,pseudorewards);
agent=DualSystemsAgent(horizons,mdp);
[agent_with_PR,R_total_with_PR,episodes_with_PR]=mdp_with_PR.simulate(agent,nr_episodes,nr_simulations)

bin_width=10;
[avg_returns_withoutPR,sem_returns_withoutPR]=binnedAverage(R_total_without_PR.mean(:),bin_width)
[avg_returns_withPR,sem_returns_withPR]=binnedAverage(R_total_with_PR.mean(:),bin_width)

nr_bins=numel(avg_returns_withoutPR);
bins=bin_width:bin_width:(bin_width*nr_bins);
fid_LC=figure()
errorbar(bins,avg_returns_withoutPR,sem_returns_withoutPR),hold on
errorbar(bins,avg_returns_withPR,sem_returns_withPR),hold on
xlim([-5,nr_episodes+5])
ylabel('Avg. Return','FontSize',18)
xlabel('Trial Number','FontSize',18)
set(gca,'FontSize',16)
legend('without PR','with PR')
%saveas(fid_LC,'MetalevelLearningCurves.fig')

for sim=1:nr_simulations
    w_with_PR(:,sim)=agent_with_PR{sim}.w_Q_meta;
    w_without_PR(:,sim)=agent_without_PR{sim}.w_Q_meta;
end

fid_Q=figure()
subplot(2,1,1)
imagesc(reshape(mean(w_with_PR,2),[2,7])),colorbar()
set(gca,'YTick',1:2,'YTickLabel',{'Habits','Planning'})
set(gca,'XTickLabel',{'Smiths-','Williams-','Jones-','Browns-','Clarks-','Bakers-','Terminal State'})
title('Meta-level Q-function learned with PRs','FontSize',18)
xlabel('State','FontSize',16)
ylabel('Planning System','FontSize',16)
subplot(2,1,2)
imagesc(reshape(mean(w_without_PR,2),[2,7])),colorbar()
set(gca,'YTick',1:2,'YTickLabel',{'Habits','Planning'})
set(gca,'XTickLabel',{'Smiths-','Williams-','Jones-','Browns-','Clarks-','Bakers-','Terminal State'})
title('Meta-level Q-function learned without PRs','FontSize',18)
xlabel('State','FontSize',16)
ylabel('Planning System','FontSize',16)
%saveas(fid_Q,'MetalevelQFunctionsIntegrated1000Trials.fig')

%% performance of planning for 10 steps
gamma=1-1/6;

transition_matrices(:,:,1) = [0,1,0,0,0,0; 0,0,1,0,0,0; 0,0,0,1,0,0; 0,0,0,0,1,0; 0,0,0,0,0,1; 1,0,0,0,0,0];
transition_matrices(:,:,2) = [0,0,0,1,0,0; 0,0,0,0,1,0; 0,0,0,0,0,1; 0,1,0,0,0,0; 1,0,0,0,0,0; 0,0,1,0,0,0];

%Rewards
small_positive=3;
small_negative=-3;
large_positive=14;
large_negative=-7;

rewards(1,:,1) = [0,large_positive,0,0,0,0];
rewards(2,:,1) = [0,0,small_negative,0,0,0];
rewards(3,:,1) = [0,0,0,small_negative,0,0];
rewards(4,:,1) = [0,0,0,0,small_negative,0];
rewards(5,:,1) = [0,0,0,0,0,small_negative];
rewards(6,:,1) = [small_negative,0,0,0,0,0];

rewards(1,:,2) = [0,0,0,small_positive,0,0];
rewards(2,:,2) = [0,0,0,0,large_negative,0];
rewards(3,:,2) = [0,0,0,0,0,large_negative];
rewards(4,:,2) = [0,small_positive,0,0,0,0];
rewards(5,:,2) = [large_negative,0,0,0,0,0];
rewards(6,:,2) = [0,0,small_positive,0,0,0];


[V, policy, cpu_time] = mdp_finite_horizon(transition_matrices, rewards, gamma, 10)

Q=NaN(6,2);

for s=1:6
    for a=1:2
        s_next=find(transition_matrices(s,:,a));
        Q(s,a)=rewards(s,s_next,a)+gamma*V(s_next,2);
    end
end

figure()
imagesc(Q'),colorbar()
set(gca,'YTick',1:2)
ylabel('Action','FontSize',16)
set(gca,'XTickLabel',{'Smiths-','Williams-','Jones-','Browns-','Clarks-','Bakers-'},'FontSize',16)
title('Q-function computed by 10-step planning','FontSize',18)

%% Simulate planning expriment with a fixed number of moves per problem: Experiment 3
clear

load('Accelerating Learning with PRs/Planning Problems/experiment3.mat')
nr_simulations=500;

mu=[1;0;-1; -1;-1;-1;-1]%[1;-1;0;1-0.25*2^1;1-0.25*2^2;1-0.25*2^3;1-0.25*2^4];
sigma=0.1;

results_with_PR=simulatePlanningExperiment(experiment3,true,nr_simulations,false,true,mu,sigma)
results_without_PR=simulatePlanningExperiment(experiment3,false,nr_simulations,false,true,mu,sigma)

mean([mean(results_with_PR.relative_horizon,2),mean(results_without_PR.relative_horizon,2)])

mean([mean(results_with_PR.relative_returns,2),mean(results_without_PR.relative_returns,2)])
mean([mean(results_with_PR.relative_returns,2),mean(results_without_PR.relative_returns,2)])


model_predictions_exp3.mean_relative_performance=mean(results_without_PR.relative_returns,2);
model_predictions_exp3.sem_relative_performance=sem(results_without_PR.relative_returns,2);
model_predictions_exp3.mean_freq_optimal_seq=mean(results_without_PR.relative_returns==1,2);
model_predictions_exp3.sem_freq_optimal_seq=sem(results_without_PR.relative_returns==1,2);

%save  Accelerating' Learning with PRs'/Planning' Problems'/model_predictions_exp3.mat model_predictions_exp3
 
figure()
subplot(2,1,1)
errorbar(1:13,mean(results_with_PR.relative_returns(1:end,:),2),sem(results_with_PR.relative_returns(1:end,:),2),'g-','LineWidth',2),hold on
errorbar(1:13,mean(results_without_PR.relative_returns(1:end,:),2),sem(results_without_PR.relative_returns(1:end,:),2),'r-','LineWidth',2)
xlim([0.8,13.2])
set(gca,'FontSize',16)
legend('With PR','Without PR')
xlabel('Trial Nr.', 'FontSize',18)
ylabel('Relative Return','FontSize',18)
subplot(2,1,2)
errorbar(4:13,mean(results_with_PR.chosen_horizons(4:end,:)==4,2),sem(results_with_PR.chosen_horizons(4:end,:)==4,2),'g-','LineWidth',2),hold on
errorbar(4:13,mean(results_without_PR.chosen_horizons(4:end,:)==4,2),sem(results_with_PR.chosen_horizons(4:end,:)==4,2),'r-','LineWidth',2),hold on
xlim([0.8,13.2])
xlabel('Trial Nr.', 'FontSize',18)
ylabel('Freq. of 4-step planning','FontSize',18)
set(gca,'FontSize',16)
legend('With PR','Without PR')

figure()
errorbar(mean(results_with_PR.chosen_horizons,2),...
    sem(results_with_PR.chosen_horizons,2),'g-','LineWidth',2),hold on
errorbar(mean(results_without_PR.chosen_horizons,2),...
    sem(results_without_PR.chosen_horizons,2),'r-','LineWidth',2)
xlim([0.8,13.2])
xlabel('Trial Nr.', 'FontSize',18)
ylabel('Planning Horizon','FontSize',18)
set(gca,'FontSize',16)
legend('With PR','Without PR')


training_problems=4:8;
test_problems=9:13;

avg_performance(:,1)=[mean(mean(results_with_PR.relative_returns(training_problems,:),2)),
mean(mean(results_with_PR.relative_returns(test_problems,:),2))];
avg_performance(:,2)=[mean(mean(results_without_PR.relative_returns(training_problems,:),2)),
mean(mean(results_without_PR.relative_returns(test_problems,:),2))]

sem_performance(:,1)=[sem(reshape(results_with_PR.relative_returns(training_problems,:),1,[])'),
sem(reshape(results_with_PR.relative_returns(test_problems,:),1,[])')];
sem_performance(:,2)=[sem(reshape(results_without_PR.relative_returns(training_problems,:),1,[])'),
sem(reshape(results_without_PR.relative_returns(test_problems,:),1,[])')];


figure()
barwitherr(sem_performance,avg_performance)
set(gca,'XTickLabel',{'Training','Transfer'},'FontSize',16)
legend('With PR','Without PR')
ylabel('Avg. Relative Returns','FontSize',16)

%% Simulate Planning Experiment 4
%clear
load experiment2
nr_simulations=500;

learn_from_PR=true;

mu=[1;0;-1; -1;-1;-1;-1]%[1;-1;0;1-0.25*2^1;1-0.25*2^2;1-0.25*2^3;1-0.25*2^4];
sigma=0.1;

results_without_PR=simulatePlanningExperiment(experiment2,false,nr_simulations,learn_from_PR,true,mu,sigma)
%results_with_PR=simulatePlanningExperiment(experiment2,true,nr_simulations,learn_from_PR,true,mu,sigma)

mean([mean(results_with_PR.relative_horizon,2),mean(results_without_PR.relative_horizon,2)])

mean([mean(results_with_PR.relative_returns,2),mean(results_without_PR.relative_returns,2)])
mean([mean(results_with_PR.relative_returns,2),mean(results_without_PR.relative_returns,2)])

figure()
subplot(2,1,1)
errorbar(mean(results_with_PR.relative_returns,2),sem(results_with_PR.relative_returns,2),'g-','LineWidth',2),hold on
errorbar(mean(results_without_PR.relative_returns,2),sem(results_without_PR.relative_returns,2),'r-','LineWidth',2)
xlim([0.8,13.2])
set(gca,'FontSize',16)
legend('With PR','Without PR')
xlabel('Trial Nr.', 'FontSize',18)
ylabel('Relative Return','FontSize',18)
subplot(2,1,2)
errorbar(mean(results_with_PR.chosen_horizons,2),...
    sem(results_with_PR.chosen_horizons,2),'g-','LineWidth',2),hold on
errorbar(mean(results_without_PR.chosen_horizons,2),...
    sem(results_without_PR.chosen_horizons,2),'r-','LineWidth',2)
xlim([0.8,13.2])
xlabel('Trial  Nr.', 'FontSize',18)
ylabel('Planning Horizon','FontSize',18)
set(gca,'FontSize',16)
legend('With PR','Without PR')

training_problems=4:8;
test_problems= 9:13;
 
avg_performance(:,1)=[mean(mean(results_with_PR.relative_returns(training_problems,:),2)),
mean(mean(results_with_PR.relative_returns(test_problems,:),2))];
avg_performance(:,2)=[mean(mean(results_without_PR.relative_returns(training_problems,:),2)),
mean(mean(results_without_PR.relative_returns(test_problems,:),2))]

model_predictions_exp4.mean_relative_performance=mean(results_with_PR.relative_returns(:,:),2);
model_predictions_exp4.sem_relative_performance=sem(results_with_PR.relative_returns(:,:),2);
model_predictions_exp4.mean_optimal_seq=mean(results_with_PR.relative_returns(:,:)==1,2);
model_predictions_exp4.sem_optimal_seq=sem(results_with_PR.relative_returns(:,:)==1,2);
%save Accelerating' Learning with PRs'/Planning' Problems'/model_predictions_exp4.mat model_predictions_exp4

sem_performance(:,1)=[sem(reshape(results_with_PR.relative_returns(training_problems,:),1,[])'),
sem(reshape(results_with_PR.relative_returns(test_problems,:),1,[])')];
sem_performance(:,2)=[sem(reshape(results_without_PR.relative_returns(training_problems,:),1,[])'),
sem(reshape(results_without_PR.relative_returns(test_problems,:),1,[])')];


figure()
barwitherr(sem_performance,avg_performance)
set(gca,'XTickLabel',{'Training','Transfer'},'FontSize',16)
legend('With PR','Without PR')
ylabel('Avg. Relative Returns','FontSize',16)

figure()
plot(4:13,mean(results_with_PR.chosen_horizons(4:end,:)==4,2),'g-','LineWidth',2),hold on
plot(4:13,mean(results_without_PR.chosen_horizons(4:end,:)==4,2),'r-','LineWidth',2),hold on
xlabel('Trial Nr.', 'FontSize',18)
ylabel('Frequency of planning 4 steps ahead','FontSize',18)
set(gca,'FontSize',16)
legend('With PR','Without PR')

%% Simulate Planning Experiment 5
clear
load Accelerating' Learning with PRs'/Planning' Problems'/experiment5.mat
nr_simulations=200;

for t=1:numel(experiment5)
    experiment5(t).states=1:6;
    experiment5(t).actions=1:2;
end

experiment.block1=experiment5(1:15);
experiment.block2=experiment5(16:18);
clear experiment5
experiment5=experiment;



learn_from_PR=true;
results_with_PR=simulatePlanningExperiment(experiment5,true,nr_simulations,learn_from_PR)
results_without_PR=simulatePlanningExperiment(experiment5,false,nr_simulations,false)

mean([mean(results_with_PR.relative_horizon,2),mean(results_without_PR.relative_horizon,2)])

mean([mean(results_with_PR.relative_returns,2),mean(results_without_PR.relative_returns,2)])
mean([mean(results_with_PR.relative_returns,2),mean(results_without_PR.relative_returns,2)])

figure()
subplot(2,1,1)
errorbar(mean(results_with_PR.relative_returns,2),sem(results_with_PR.relative_returns,2),'g-','LineWidth',2),hold on
errorbar(mean(results_without_PR.relative_returns,2),sem(results_without_PR.relative_returns,2),'r-','LineWidth',2)
xlim([0.8,13.2])
set(gca,'FontSize',16)
legend('With PR','Without PR')
xlabel('Trial Nr.', 'FontSize',18)
ylabel('Relative Return','FontSize',18)
subplot(2,1,2)
errorbar(mean(results_with_PR.chosen_horizons,2),...
    sem(results_with_PR.chosen_horizons,2),'g-','LineWidth',2),hold on
errorbar(mean(results_without_PR.chosen_horizons,2),...
    sem(results_without_PR.chosen_horizons,2),'r-','LineWidth',2)
xlim([0.8,18.2])
xlabel('Trial Nr.', 'FontSize',18)
ylabel('Planning Horizon','FontSize',18)
set(gca,'FontSize',16)
legend('With PR','Without PR')

training_problems=1:15;
test_problems=16:18;

avg_performance(:,1)=[mean(mean(results_with_PR.relative_returns(training_problems,:),2)),
mean(mean(results_with_PR.relative_returns(test_problems,:),2))];
avg_performance(:,2)=[mean(mean(results_without_PR.relative_returns(training_problems,:),2)),
mean(mean(results_without_PR.relative_returns(test_problems,:),2))]

sem_performance(:,1)=[sem(reshape(results_with_PR.relative_returns(training_problems,:),1,[])'),
sem(reshape(results_with_PR.relative_returns(test_problems,:),1,[])')];
sem_performance(:,2)=[sem(reshape(results_without_PR.relative_returns(training_problems,:),1,[])'),
sem(reshape(results_without_PR.relative_returns(test_problems,:),1,[])')];


figure()
barwitherr(sem_performance,avg_performance)
set(gca,'XTickLabel',{'Training','Transfer'},'FontSize',16)
legend('With PR','Without PR')
ylabel('Avg. Relative Returns','FontSize',16)

nr_trials=18;
planning_horizons=[experiment5.block1.horizon,experiment5.block2.horizon]';
figure()
plot(1:nr_trials,mean(results_with_PR.chosen_horizons(1:nr_trials,:)==repmat(planning_horizons,[1,nr_simulations]),2),'g-','LineWidth',2),hold on
plot(1:nr_trials,mean(results_without_PR.chosen_horizons(1:nr_trials,:)==repmat(planning_horizons,[1,nr_simulations]),2),'r-','LineWidth',2),hold on
xlabel('Trial Nr.', 'FontSize',18)
ylabel('Frequency of optimal planning','FontSize',18)
set(gca,'FontSize',16)
legend('With PR','Without PR')

%% Simulate Experiment 6
clear
load Accelerating' Learning with PRs'/Planning' Problems'/experiment6.mat
nr_simulations=1000;

for t=1:numel(experiment6)
    experiment6(t).states=1:6;
    experiment6(t).actions=1:2;
end

experiment.block1=experiment6(1:16);
experiment.block2=experiment6(17:20);
clear experiment6
experiment6=experiment;

mu=[1;0;-1; -1;-1;-1;-1]%[1;-1;0;1-0.25*2^1;1-0.25*2^2;1-0.25*2^3;1-0.25*2^4];
sigma=0.1;

learn_from_PR=true;
immediate_feedback=false;
results_with_PR=simulatePlanningExperiment(experiment6,true,nr_simulations,learn_from_PR,immediate_feedback,mu,sigma);
results_without_PR=simulatePlanningExperiment(experiment6,false,nr_simulations,false,immediate_feedback,mu,sigma);

mean([mean(results_with_PR.relative_horizon,2),mean(results_without_PR.relative_horizon,2)])

mean([mean(results_with_PR.relative_returns,2),mean(results_without_PR.relative_returns,2)])
mean([mean(results_with_PR.relative_returns,2),mean(results_without_PR.relative_returns,2)])

figure()
subplot(2,1,1)
errorbar(mean(results_with_PR.relative_returns,2),sem(results_with_PR.relative_returns,2),'g-','LineWidth',2),hold on
errorbar(mean(results_without_PR.relative_returns,2),sem(results_without_PR.relative_returns,2),'r-','LineWidth',2)
xlim([0.8,20.2])
set(gca,'FontSize',16)
legend('With FB','Without FB')
xlabel('Trial Nr.', 'FontSize',18)
ylabel('Relative Return','FontSize',18)
subplot(2,1,2)
errorbar(mean(results_with_PR.chosen_horizons,2),...
    sem(results_with_PR.chosen_horizons,2),'g-','LineWidth',2),hold on
errorbar(mean(results_without_PR.chosen_horizons,2),...
    sem(results_without_PR.chosen_horizons,2),'r-','LineWidth',2)
xlim([0.8,20.2])
xlabel('Trial Nr.', 'FontSize',18)
ylabel('Planning Horizon','FontSize',18)
set(gca,'FontSize',16)
legend('With PR','Without PR')

training_problems=1:16;
test_problems=17:20;

avg_performance(:,1)=[mean(mean(results_with_PR.relative_returns(training_problems,:),2)),
mean(mean(results_with_PR.relative_returns(test_problems,:),2))];
avg_performance(:,2)=[mean(mean(results_without_PR.relative_returns(training_problems,:),2)),
mean(mean(results_without_PR.relative_returns(test_problems,:),2))]

sem_performance(:,1)=[sem(reshape(results_with_PR.relative_returns(training_problems,:),1,[])'),
sem(reshape(results_with_PR.relative_returns(test_problems,:),1,[])')];
sem_performance(:,2)=[sem(reshape(results_without_PR.relative_returns(training_problems,:),1,[])'),
sem(reshape(results_without_PR.relative_returns(test_problems,:),1,[])')];


figure()
barwitherr(sem_performance,avg_performance)
set(gca,'XTickLabel',{'Training','Transfer'},'FontSize',16)
legend('With PR','Without PR')
ylabel('Avg. Relative Returns','FontSize',16)

nr_trials=20;
planning_horizons=[experiment6.block1.horizon,experiment6.block2.horizon]';
figure()
plot(1:nr_trials,mean(results_with_PR.chosen_horizons(1:nr_trials,:)==repmat(planning_horizons,[1,nr_simulations]),2),'g-','LineWidth',2),hold on
plot(1:nr_trials,mean(results_without_PR.chosen_horizons(1:nr_trials,:)==repmat(planning_horizons,[1,nr_simulations]),2),'r-','LineWidth',2),hold on
xlabel('Trial Nr.', 'FontSize',18)
ylabel('Frequency of optimal planning','FontSize',18)
set(gca,'FontSize',16)
legend('With PR','Without PR')


model_predictions_exp6(1).mean_relative_performance=mean(results_without_PR.relative_returns(1:nr_trials,:),2);
model_predictions_exp6(2).mean_relative_performance=mean(results_with_PR.relative_returns(1:nr_trials,:),2);
model_predictions_exp6(1).sem_relative_performance=sem(results_without_PR.relative_returns(1:nr_trials,:)')';
model_predictions_exp6(2).sem_relative_performance=sem(results_with_PR.relative_returns(1:nr_trials,:)')';

model_predictions_exp6(1).mean_freq_optimal_seq=mean(results_without_PR.relative_returns(1:nr_trials,:)==1,2);
model_predictions_exp6(2).mean_freq_optimal_seq=mean(results_with_PR.relative_returns(1:nr_trials,:)==1,2);
model_predictions_exp6(1).sem_freq_optimal_seq=sem(results_without_PR.relative_returns(1:nr_trials,:)'==1)';
model_predictions_exp6(2).sem_freq_optimal_seq=sem(results_with_PR.relative_returns(1:nr_trials,:)'==1)';

%save Accelerating' Learning with PRs'/Planning' Problems'/model_predictions_exp6.mat model_predictions_exp6