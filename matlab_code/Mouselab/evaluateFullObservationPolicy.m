function evaluateFullObservationPolicy(c)
%% benchmark policy: observing everything before the first move

rng('shuffle')

addpath('/global/home/users/flieder/matlab_code/MatlabTools/')
%create meta-level MDP


add_pseudorewards=false;
pseudoreward_type='none';

mean_payoff=4.5;
std_payoff=10.6;

load('/global/home/users/flieder/matlab_code/MouselabMDPExperiment_normalized')

meta_MDP=MouselabMDPMetaMDPNIPS(add_pseudorewards,pseudoreward_type,mean_payoff,std_payoff,experiment);
meta_MDP.cost_per_click=c;

policy=@(state,mdp) fullObservationPolicy(state,mdp)

nr_episodes_evaluation=2000;%2000;
[R_total,problems,states,chosen_actions,indices]=...
    inspectPolicyGeneral(meta_MDP,policy,nr_episodes_evaluation)

reward_full_observation_policy=[mean(R_total),sem(R_total)];
nr_observations_full_observation_policy=...
    [mean(indices.nr_acquisitions),sem(indices.nr_acquisitions(:))];

%full_observation_benchmark.policy=policy;
full_observation_benchmark.reward=reward_full_observation_policy;
full_observation_benchmark.nr_observations=nr_observations_full_observation_policy;
full_observation_benchmark.returns=R_total;
full_observation_benchmark.cost_per_click=c;

save(['/global/home/users/flieder/results/','full_observation_benchmark',...
    int2str(round(100*c)),'.mat'],'full_observation_benchmark','-v7.3')

end