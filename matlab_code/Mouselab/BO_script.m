addpath('../MatlabTools/')
addpath('/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/matlab_code/Mouselab/')
load low_cost_condition
load medium_cost_condition
load high_cost_condition

%scaling_factor = 2*16/12;
scaling_factor=1;
low_costs=ceil(100*[0.01]/scaling_factor)/100; %[0.01]
edmedium_costs=ceil(100*[1]/scaling_factor)/100;%[0.25,0.50,0.75,1];
high_costs=ceil(100*[2.5]/scaling_factor)/100;%[1.50,2,2.50,3];

costs=round(100*[0.01,1,2.50]/scaling_factor)/100;
conditions={low_cost_condition,medium_cost_condition,high_cost_condition};
names={'low_cost_VPIallActions','medium_cost_VPIallActions','high_cost_VPIallActions'};

parfor c=1:numel(conditions)
    policySearchMouselabMDP(costs(c),conditions{c},names{c})
end

%{
test_costs=[0,0.00375,0.725,0.775];%[0.01,0.25,0.50,0.55,0.60,0.65,0.70,0.75];%[0.01, 0.25, 0.50, 1.25, 1.50];
parfor c=1:numel(test_costs)
    policySearchMouselabMDP(test_costs(c),low_cost_condition,'test_cost')
end
%}

%{
parfor c=1:numel(low_costs)
    policySearchMouselabMDP(low_costs(c),low_cost_condition,'low_cost')
end

parfor c=1:numel(medium_costs)
    policySearchMouselabMDP(medium_costs(c),medium_cost_condition,'medium_cost')
end

parfor c=1:numel(high_costs)
    policySearchMouselabMDP(high_costs(c),high_cost_condition,'high_cost')
end
%}
%%
clear

addpath('../MatlabTools/')
addpath('/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/matlab_code/Mouselab/')
%load low_cost_condition
%load medium_cost_condition
%load high_cost_condition

%condition='lowCost';

addpath([pwd,'../MatlabTools/'])
%create meta-level MDP

add_pseudorewards=false;
pseudoreward_type='none';


conditions={'lowCost','mediumCost','highCost'};

for c=1:numel(conditions)
    
    condition=conditions{c};
    
    if strcmp(condition,'highCost')
        %load high_cost_condition
        %experiment = high_cost_condition;
        cost = 2.50;
        
        temp=load(['../../results/BO/BO_c250n100high_cost_VPIallActions.mat']);
        w_policy=temp.BO.w_hat;
        
        training_data1=load('~/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/0.6/stateActions_high_none.mat');
        training_data2=load('~/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/0.6/stateActions_high_none.mat');
        %training_data2=load('~/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/0.6/stateActions_high_featureBased.mat');
        load('~/Dropbox/PhD/Metacognitive RL/MCRL/matlab_code/Mouselab/high_cost_condition_pilot.mat')
        experiment=high_cost_condition;
    end
    
    if strcmp(condition,'mediumCost')
        %load medium_cost_condition
        %experiment = medium_cost_condition;
        cost = 1.00;
        
        temp=load(['../../results/BO/BO_c100n100medium_cost_VPIallActions.mat']);
        w_policy=temp.BO.w_hat;
        
        training_data1=load('~/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/0.6/stateActions_medium_none.mat');
        training_data2=load('~/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/0.6/stateActions_medium_none.mat');
        %training_data2=load('~/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/0.6/stateActions_medium_featureBased.mat');
        %training_data2=load('~/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/0.6/stateActions_low_none.mat');
        load('~/Dropbox/PhD/Metacognitive RL/MCRL/matlab_code/Mouselab/medium_cost_condition_pilot.mat')
        experiment=medium_cost_condition;
    end
    
    if strcmp(condition,'lowCost')
        %load low_cost_condition
        %experiment = low_cost_condition;
        cost=0.01;
        
        temp=load(['../../results/BO/BO_c1n100low_cost_VPIallActions.mat']);
        w_policy=temp.BO.w_hat;
        
        training_data1=load('~/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/0.6/stateActions_low_none.mat')
        training_data2=load('~/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/0.6/stateActions_low_none.mat')
        %training_data2=load('~/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/0.6/stateActions_low_featureBased.mat')
        
        load('~/Dropbox/PhD/Metacognitive RL/MCRL/matlab_code/Mouselab/low_cost_condition_pilot.mat')
        experiment=low_cost_condition;
    end
    
    for t=1:numel(experiment)
        avg_payoff(t)=nanmean(experiment(t).rewards(experiment(t).T>0));
        stds_payoff(t)=nanstd(experiment(t).rewards(experiment(t).T>0));
    end
    mean_payoff=mean(avg_payoff);
    std_payoff=mean(stds_payoff);
               
    training_data.trialID=[training_data1.trialID,training_data2.trialID];
    training_data.trialNr=[training_data1.trialNr,training_data2.trialNr];
    training_data.state_actions=[training_data1.state_actions,training_data2.state_actions];
    training_data.rewardSeen=[training_data1.rewardSeen,training_data2.rewardSeen];
    training_data.trials=experiment;    
    
    %costs=[0.01,1.60,2.80];
    %w_observe_everything=[1;1;1];
    %w_observe_nothing=[-1.3;0.4;0.3];
    
    %[ER_hat_everything,result_everything]=evaluatePolicy(w_observe_everything,0.01,1000)
    %[ER_hat_nothing,result_nothing]=evaluatePolicy(w_observe_nothing,2.80,1000)
    
    nr_episodes=numel(training_data.trialNr);
    %parfor c=1:numel(costs)
    
    glm_policy=BayesianGLM(numel(w_policy),1e-6);
    glm_policy.mu_n=w_policy;
    glm_policy.mu_0=w_policy;
    
    meta_MDP=MouselabMDPMetaMDPNIPS(add_pseudorewards,pseudoreward_type,mean_payoff,std_payoff,experiment);
    meta_MDP.cost_per_click=cost;
    
    feature_extractor=@(s,c,meta_MDP) meta_MDP.QFeatures(s,c);
    
    [glm_Q(c),MSE{c},R_total{c},F_history{c},R_history{c},s_history{c},c_history{c}]=...
        BayesianValueFunctionRegression(meta_MDP,feature_extractor,nr_episodes,glm_policy,training_data);
end
%end
fit.glm=glm_Q; fit.MSE=MSE; fit.R_total=R_total; fit.conditions=conditions;
fit.featureNames={'is_click*(E[max_a Q(s,a)]-max_a)','VPI', 'VOC+cost', 'cost', 'ER_act','1'};
save ../../results/BO/valueFunctionFitConditionSpecificTrainingDataFromControlCondition.mat fit