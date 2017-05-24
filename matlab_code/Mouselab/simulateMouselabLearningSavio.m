function simulateMouselabLearningSavio(repetition,condition,with_PR)

addpath('/global/home/users/flieder/matlab_code/MatlabTools/')
%create meta-level MDP

add_pseudorewards=true;
pseudoreward_type='featureBased';

mean_payoff=4.5;
std_payoff=10.6;

weights_low_cost.VPI=1.8668;
weights_low_cost.VOC1=-0.2205;
weights_low_cost.ER=1.0034;

weights_medium_cost.VPI=0.6979;
weights_medium_cost.VOC1=-0.0840;
weights_medium_cost.ER=0.9584;

weights_high_cost.VPI=0.3171;
weights_high_cost.VOC1=-0.1872;
weights_high_cost.ER=0.9375;

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

meta_MDP_with_PR=MouselabMDPMetaMDPNIPS(add_pseudorewards,pseudoreward_type,mean_payoff,std_payoff,experiment);

meta_MDP_without_PR=MouselabMDPMetaMDPNIPS(false,'',mean_payoff,std_payoff,experiment);

switch condition
    case 1
        meta_MDP_with_PR.cost_per_click=0.01;
        meta_MDP_with_PR.PR_feature_weights=weights_low_cost;        
        
        meta_MDP_without_PR.cost_per_click=0.01;        
    case 2
        meta_MDP_with_PR.cost_per_click=1.60;
        meta_MDP_with_PR.PR_feature_weights=weights_medium_cost;             
        
        meta_MDP_without_PR.cost_per_click=1.60;
    case 3
        meta_MDP_with_PR.cost_per_click=2.80;
        meta_MDP_with_PR.PR_feature_weights=weights_high_cost;
        
        meta_MDP_without_PR.cost_per_click=2.80;
end

mu0=[1;1;1];
nr_features=numel(mu0);
sigma0=0.1;
glm0=BayesianGLM(nr_features,sigma0);
glm0.mu_n=mu0(:);

feature_extractor=@(s,c,meta_MDP) meta_MDP.extractStateActionFeatures(s,c);

%load MouselabMDPMetaMDPTestFeb-17-2017

nr_training_episodes=1000;
 nr_reps=1;
first_episode=1; last_rep=nr_training_episodes;
for rep=1:nr_reps
    
    glm(rep)=glm0;
    
    if with_PR
        tic()
        [glm_with_PR(rep),MSE_with_PR(first_episode:nr_training_episodes,rep),...
            returns_with_PR(first_episode:nr_training_episodes,rep)]=BayesianSARSAQ(...
            meta_MDP_with_PR,feature_extractor,nr_training_episodes-first_episode+1,glm(rep));
        disp(['Repetition ',int2str(rep),' took ',int2str(round(toc()/60)),' minutes.'])
    else
        [glm_without_PR(rep),MSE_without_PR(first_episode:nr_training_episodes,rep),...
            returns_without_PR(first_episode:nr_training_episodes,rep)]=BayesianSARSAQ(...
            meta_MDP_without_PR,feature_extractor,nr_training_episodes-first_episode+1,glm(rep));
    end
end

if with_PR
    result.avg_return_with_PR=[mean(returns_with_PR(:)),sem(returns_with_PR(:))];
    result.returns_with_PR=returns_with_PR;
    save(['/global/home/users/flieder/results/learning/MouselabLearningWithPR_rep',int2str(repetition),'_condition',int2str(condition),'.mat'],'result')
else
    result.avg_return_without_PR=[mean(returns_without_PR(:)),sem(returns_without_PR(:))];
    result.returns_without_PR=returns_without_PR;
    save(['/global/home/users/flieder/results/learning/MouselabLearningWithoutPR_rep',int2str(repetition),'_condition',int2str(condition),'.mat'],'result')
end


end