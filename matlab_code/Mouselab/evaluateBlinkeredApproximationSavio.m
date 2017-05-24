function evaluateBlinkeredApproximationSavio(cost)

addpath(genpath('/global/home/users/flieder/matlab_code/MatlabTools/'))
%create meta-level MDP

add_pseudorewards=false;
pseudoreward_type='none';

mean_payoff=4.5;
std_payoff=10.6;

load('/global/home/users/flieder/matlab_code/MouselabMDPExperiment_normalized')

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
meta_MDP.cost_per_click=cost;
meta_MDP.object_level_MDP=meta_MDP.object_level_MDPs(1);

[state,meta_MDP]=sampleS0(meta_MDP);

meta_MDP=meta_MDP.computeBlinkeredPolicy(state);

%{
%initial testing

delta_mu=-0.1;
sigma_mu=10;
observed=[0,1,0,0];
c=1;
Q_hat=meta_MDP.getQBlinkered(delta_mu,sigma_mu,observed,c)
c_blinkered=meta_MDP.piBlinkered(state)
%}

blinkered_policy = @(state,meta_mdp) meta_mdp.piBlinkered(state);

[R_total_blinkered,problems,states,chosen_actions,indices]=...
    inspectPolicyGeneral(meta_MDP,blinkered_policy,10000);


result.ER=[mean(R_total_blinkered),sem(R_total_blinkered)];
result.nr_computations=[mean(indices.nr_acquisitions),sem(indices.nr_acquisitions(:))];

save(['/global/home/users/flieder/results/performance_blinkered_approximation_c',int2str(100*cost)],'result')

end