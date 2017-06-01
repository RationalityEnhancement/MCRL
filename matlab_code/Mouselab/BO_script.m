addpath('../MatlabTools/')

costs=[0.01,1.60,2.80]; %,1.60 is already taken care of

parfor c=1:numel(costs)
    policySearchMouselabMDP(costs(c))
end

%%
clear

addpath([pwd,'/MatlabTools/'])
%create meta-level MDP

add_pseudorewards=false;
pseudoreward_type='none';

mean_payoff=4.5;
std_payoff=10.6;

load ControlExperiment

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
for e=1:numel(control_experiment)
    control_experiment(e).actions_by_state=actions_by_state;
    control_experiment(e).hallway_states=2:9;
    control_experiment(e).leafs=10:17;
    control_experiment(e).parent_by_state={1,1,1,1,1,2,3,4,5,6,6,7,7,8,8,9,9};
end

training_data1=load('~/Dropbox/PhD/Metacognitive RL/mcrl-experiment/1E_state-action_pairs/stateActions_0.mat');
training_data2=load('~/Dropbox/PhD/Metacognitive RL/mcrl-experiment/1E_state-action_pairs/stateActions_2.mat');

training_data.trialID=[training_data1.trialID,training_data2.trialID];
training_data.trialNr=[training_data1.trialNr,training_data2.trialNr];
training_data.state_actions=[training_data1.state_actions,training_data2.state_actions];
training_data.rewardSeen=[training_data1.rewardSeen,training_data2.rewardSeen];
training_data.trials=control_experiment;

costs=[0.01,1.60,2.80];
%w_observe_everything=[1;1;1];
%w_observe_nothing=[-1.3;0.4;0.3];

%[ER_hat_everything,result_everything]=evaluatePolicy(w_observe_everything,0.01,1000)
%[ER_hat_nothing,result_nothing]=evaluatePolicy(w_observe_nothing,2.80,1000)

nr_episodes=numel(training_data.trialNr);
parfor c=1:numel(costs)
    
    temp=load(['../../results/BO/BO_c',int2str(100*costs(c)),'n35.mat'])
    
    w_policy=temp.BO.w_hat;
    glm_policy=BayesianGLM(3,1e-6);
    glm_policy.mu_n=w_policy;
    glm_policy.mu_0=w_policy;    
    
    meta_MDP=MouselabMDPMetaMDPNIPS(add_pseudorewards,pseudoreward_type,mean_payoff,std_payoff,control_experiment);
    meta_MDP.cost_per_click=costs(c);
    
    feature_extractor=@(s,c,meta_MDP) [meta_MDP.extractStateActionFeatures(s,c); c.is_computation*meta_MDP.cost_per_click];
    
    [glm_Q(c),MSE(:,c),R_total(:,c)]=BayesianValueFunctionRegression(meta_MDP,feature_extractor,nr_episodes,glm_policy,training_data)
    
end
fit.glm=glm_Q; fit.MSE=MSE; fit.R_total=R_total;
save ../../results/BO/valueFunctionFit.mat fit

%% 
costs=[0.01,1.60,2.80];
nr_episodes_evaluation=1000;
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

for c=1:3
    meta_MDP=MouselabMDPMetaMDPNIPS(false,'none',4.5,10.6,experiment,costs(c));
    
    temp=load(['../../results/BO/BO_c',int2str(100*costs(c)),'n35.mat']);
    
    policy=@(state,mdp) deterministicPolicy(state,mdp,temp.BO.w_hat);
    
    for t=1:nr_trials
        [R_total,problems{t,c},states{t,c},chosen_actions{t,c},indices(t,c)]=inspectPolicyGeneral(meta_MDP,policy,...
            nr_episodes_evaluation,experiment,t);
        
        ER_hat(t,c)=mean(R_total);
    end
end

for c=1:3
    for t=1:nr_trials
        avg_nr_observations(t,c)=mean(indices(t,c).nr_acquisitions);
    end
end

csvwrite('/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/score_pi_star.csv',ER_hat)
csvwrite('/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/nr_observations_pi_star.csv',avg_nr_observations)