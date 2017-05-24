addpath([pwd,'/MatlabTools/'])
%create meta-level MDP

add_pseudorewards=true;
pseudoreward_type='featureBased';

mean_payoff=4.5;
std_payoff=10.6;

weights_low_cost.VPI=1.2065;
weights_low_cost.VOC1=2.1510;
weights_low_cost.ER=1.5298;

weights_medium_cost.VPI=0.6118;
weights_medium_cost.VOC1=1.2708;
weights_medium_cost.ER=1.3215;

weights_high_cost.VPI=0.6779;
weights_high_cost.VOC1=0.7060;
weights_high_cost.ER=1.2655;

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

costs=[0.01,0.05,0.10,0.20,0.40,0.80,1.60];
c=1;

meta_MDP_with_PR=MouselabMDPMetaMDPNIPS(add_pseudorewards,pseudoreward_type,mean_payoff,std_payoff,experiment);
meta_MDP_with_PR.cost_per_click=costs(c);
meta_MDP_with_PR.PR_feature_weights=weights_low_cost;

meta_MDP_without_PR=MouselabMDPMetaMDPNIPS(false,'',mean_payoff,std_payoff,experiment);
meta_MDP_without_PR.cost_per_click=costs(c);

mu0=[1;1;1];
nr_features=numel(mu0);
sigma0=0.1;
glm0=BayesianGLM(nr_features,sigma0);
glm0.mu_n=mu0(:);

feature_extractor=@(s,c,meta_MDP) meta_MDP.extractStateActionFeatures(s,c);

%load MouselabMDPMetaMDPTestFeb-17-2017

nr_training_episodes=2000;
nr_reps=1;
first_episode=1; last_rep=nr_training_episodes;
for rep=1:nr_reps
    glm(rep)=glm0;
    tic()
    [glm_with_PR(rep),MSE_with_PR(first_episode:nr_training_episodes,rep),...
        returns_with_PR(first_episode:nr_training_episodes,rep)]=BayesianSARSAQ(...
        meta_MDP_with_PR,feature_extractor,nr_training_episodes-first_episode+1,glm(rep));
    disp(['Repetition ',int2str(rep),' took ',int2str(round(toc()/60)),' minutes.'])
    
    [glm_without_PR(rep),MSE_without_PR(first_episode:nr_training_episodes,rep),...
        returns_without_PR(first_episode:nr_training_episodes,rep)]=BayesianSARSAQ(...
        meta_MDP_without_PR,feature_extractor,nr_training_episodes-first_episode+1,glm(rep));

end

[mean(returns_with_PR(:)),sem(returns_with_PR(:))]
[mean(returns_without_PR(:)),sem(returns_without_PR(:))]

figure()
plot(smooth(returns_with_PR,100)),hold on
plot(smooth(returns_without_PR,100))