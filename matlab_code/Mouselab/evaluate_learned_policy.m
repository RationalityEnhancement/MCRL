%% Evaluate performance of learned and no-observation policy on the planning problems posed in our experiment

costs=[0.01,1.60,2.80];
nr_episodes_evaluation=1;
addpath('../MatlabTools/')

load PilotExperiment
nr_trials=numel(experiment);

actions_by_state{1}=[];
actions_by_state{2}=[1];
actions_by_state{3}=[2];
actions_by_state{4}=[3];
actions_by_state{5}=[4];
actions_by_state{6}=[1,1];
actions_by_state{7}=[2,2];
actions_by_state{8}=[3,3];
actions_by_state{9}=[4,4];
actions_by_state{10}=[1,1,2];
actions_by_state{11}=[1,1,4];
actions_by_state{12}=[2,2,3];
actions_by_state{13}=[2,2,4];
actions_by_state{14}=[3,3,2];
actions_by_state{15}=[3,3,4];
actions_by_state{16}=[4,4,3];
actions_by_state{17}=[4,4,1];
for e=1:numel(experiment)
    experiment(e).actions_by_state=actions_by_state;
    experiment(e).hallway_states=2:9;
    experiment(e).leafs=10:17;
    experiment(e).parent_by_state={1,1,1,1,1,2,3,4,5,6,6,7,7,8,8,9,9};
end
save PilotExperiment experiment

for c=1:3
    meta_MDP=MouselabMDPMetaMDPNIPS(false,'none',4.5,10.6,experiment,costs(c));
    
    temp=load(['../../results/BO/BO_c',int2str(100*costs(c)),'n35.mat']);
    
    learned_policy=@(state,mdp) deterministicPolicy(state,mdp,temp.BO.w_hat);
    no_observation_policy= @(state,mdp) noObservationPolicy(state,mdp);
    
    for t=1:nr_trials
        [R_total,problems{t,c},states{t,c},chosen_actions{t,c},indices(t,c)]=...
            inspectPolicyGeneral(meta_MDP,learned_policy,...
            nr_episodes_evaluation,experiment,t);
        
        [R_total_no_obs(t,c),problems_no_obs{t,c},states_no_obs{t,c},chosen_actions_no_obs{t,c},indices_no_obs(t,c)]=...
            inspectPolicyGeneral(meta_MDP,no_observation_policy,...
            nr_episodes_evaluation,experiment,t);

        
        ER_hat(t,c)=mean(R_total);
    end
end

for c=1:3
    for t=1:nr_trials
        avg_nr_observations(t,c)=mean(indices(t,c).nr_acquisitions);
    end
end

score_pi_no_obs = R_total_no_obs;

csvwrite('/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/score_pi_star.csv',ER_hat)
csvwrite('/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/nr_observations_pi_star.csv',avg_nr_observations)
csvwrite('/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/score_pi_no_obs.csv',R_total_no_obs)

%% Evaluate performance of no-observation policy across all environments
costs=[0.01,1.60,2.80];
nr_episodes_evaluation=1000;
addpath('../MatlabTools/')

load PilotExperiment

for c=1:3
    meta_MDP=MouselabMDPMetaMDPNIPS(false,'none',4.5,10.6,experiment,costs(c));
    
    temp=load(['../../results/BO/BO_c',int2str(100*costs(c)),'n35.mat']);
    
    no_observation_policy= @(state,mdp) noObservationPolicy(state,mdp);
    
    
    [R_total_no_obs,problems{c},states{c},chosen_actions{c},indices(c)]=...
        inspectPolicyGeneral(meta_MDP,no_observation_policy,...
        nr_episodes_evaluation,experiment);
    
    ER_hat_no_obs(c)=mean(R_total_no_obs);
    
end


for c=1:3
    temp=load(['../../results/BO/BO_c',int2str(100*costs(c)),'n35.mat'])
    ER_learned_policy(c)=temp.BO.ER;
end

[ER_learned_policy; ER_hat_no_obs]

%% 
costs=[0.01,1.6,2.8];
load('/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/score_pi_star.csv')
load('/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/score_pi_no_obs.csv')
load('/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/worst.csv')
load('/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/optimal.csv')

relative_performace_pi_star=(score_pi_star-repmat(worst',[1,3]))./(repmat(optimal,[1,3])-repmat(worst',[1,3]))
mean(relative_performace_pi_star)

relative_performance_pi_no_obs=(score_pi_no_obs-repmat(worst',[1,3]))./(repmat(optimal,[1,3])-repmat(worst',[1,3]))
mean(relative_performance_pi_no_obs)

fig=figure()
bar(mean(relative_performace_pi_star))
ylabel('Avg. Relative Performance of \pi*','FontSize',16)
set(gca,'XTickLabel',costs,'FontSize',16)
xlabel('Cost per click','FontSize',16)

saveas(fig,'/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/results/RelativeScorePiStar.png')

[mean(relative_performace_pi_star);mean(relative_performance_pi_no_obs)]

%% 
clear

addpath('..')
addpath('../MatlabTools/')

root_dir='~/Dropbox/PhD/Metacognitive RL/MCRL/';
nb_iter=[100];

conditions={'lowCost','mediumCost','highCost'};

max_score=csvread('/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/optimal.csv');
min_score=csvread('/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/worst.csv');


for c=1:length(conditions)
    
    condition=conditions{c};
    
    if strcmp(condition,'lowCost')
        root_dir='~/Dropbox/PhD/Metacognitive RL/MCRL/'
        name='low_cost';%'learnedOnTestDistributionWithCost';
        load low_cost_condition
        %costs=[0.01];
        cost=0.01;
        experiment = low_cost_condition;
    end
    
    if strcmp(condition,'mediumCost')
        
        name='medium_cost';%'learnedOnTestDistributionWithCost';
        load medium_cost_condition
        %costs=[0.25,0.50,0.75,1];
        cost=1.00;
        experiment = medium_cost_condition;
    end
    
    if strcmp(condition,'highCost')
        
        name='high_cost';%'learnedOnTestDistributionWithCost';
        load high_cost_condition
        costs=[1.50,2,2.50,3];
        cost=2.50;
        experiment = high_cost_condition;
    end
    
    nr_trials=numel(experiment);
    nr_episodes_evaluation=1;
    
    %costs=[0.01,0.25,0.50,0.75,1,1.2,1.4:0.1:1.8,2,2.5,3.0];%[0.01,0.25,0.50,0.75,1:0.1:1.4,1.60,2.80];
    %for c=1:length(costs)
    %    cost=costs(c);
    try
        path_to_file=[root_dir,'results/BO/BO_c',int2str(100*cost),'n',int2str(nb_iter(1)),name,'_VPIallActions.mat'] %_VPIallActions
        temp=load(path_to_file)
    catch
        path_to_file=[root_dir,'results/BO/BO_c',int2str(100*cost),'n',int2str(nb_iter(2)),name,'_VPIallActions.mat'] %_VPIallActions
        temp=load(path_to_file)
    end
    
    policy_weights_by_cost(:,c)=temp.BO.w_hat
    ER_policy_by_cost(c)=temp.BO.ER
    
    meta_MDP=MouselabMDPMetaMDPNIPS(false,'none',4.5,10.6,experiment,cost);
    
    learned_policy=@(state,mdp) deterministicPolicy(state,mdp,policy_weights_by_cost(:,c));
    
    for t=1:nr_trials
        [result.R_total(t,c),result.problems{t,c},result.states{t,c},result.chosen_actions{t,c},indices(t,c)]=...
            inspectPolicyGeneral(meta_MDP,learned_policy,...
            nr_episodes_evaluation,experiment,t);
        
        result.time_cost(t,c)=cost;
        
        nr_observations(t,c)=mean(indices(t,c).nr_acquisitions);
    end
    %ER_policy_by_cost(c)=mean(R_total)    
    rel_ER_by_cost(c)=mean( (result.R_total(:,c)-min_score(:,c))./(max_score(:,c)-min_score(:,c)));
    
end

nr_observations_by_cost=mean(nr_observations)

csvwrite('/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/score_pi_star.csv',ER_policy_by_cost)
csvwrite('/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/rel_score_pi_star.csv',rel_ER_by_cost)
csvwrite('/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/nr_observations_pi_star.csv',nr_observations_by_cost)

save('/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/results/behavior_of_learned_policy.mat','result') 