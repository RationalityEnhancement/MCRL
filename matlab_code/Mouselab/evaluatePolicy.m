function [ER_hat,result]=evaluatePolicy(w,c,nr_episodes_evaluation)

rng('shuffle')

%addpath('MatlabTools')

%load('/global/home/users/flieder/results/BSARSA_results_Mouselab.mat')
%c=BSARSA_results.costs(c_index);

%create meta-level MDP
glm=BayesianGLM(numel(w),0.000001);
glm.mu_0=w(:);
glm.mu_n=w(:);

add_pseudorewards=false;
pseudoreward_type='none';

mean_payoff=4.5;
std_payoff=10.6;

load('MouselabMDPExperiment_normalized')

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
    experiment(e).parent_by_state=[1,1,1,1,1,2,3,4,5,6,6,7,7,8,8,9,9];
end


meta_MDP=MouselabMDPMetaMDPNIPS(add_pseudorewards,pseudoreward_type,mean_payoff,std_payoff,experiment);
meta_MDP.cost_per_click=c;

%nr_episodes_evaluation=1000;%2000;
meta_MDP.object_level_MDP=meta_MDP.object_level_MDPs(1);
%policy=@(state,mdp) contextualThompsonSampling(state,meta_MDP,glm);
policy=@(state,mdp) deterministicPolicy(state,meta_MDP,w);
[R_total_evaluation,problems_evaluation,states_evaluation,chosen_actions_evaluation,indices_evaluation]=...
    inspectPolicyGeneral(meta_MDP,policy,nr_episodes_evaluation);

reward_learned_policy=[mean(R_total_evaluation),sem(R_total_evaluation)];
nr_observations_learned_policy=[mean(indices_evaluation.nr_acquisitions),...
    sem(indices_evaluation.nr_acquisitions(:))];

%result.policy=policy;
result.reward=reward_learned_policy;
result.weights=glm.mu_n;
result.features={'VPI','VOC','E[R|act,b]'};
result.nr_observations=nr_observations_learned_policy;
result.returns=R_total_evaluation;
result.cost_per_click=c;
result.glm=glm;

ER_hat=reward_learned_policy(1);

end