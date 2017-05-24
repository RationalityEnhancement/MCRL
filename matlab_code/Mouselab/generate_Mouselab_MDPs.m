%% Generate planning problems for Mouselab MDP experiment

nr_steps=3;
nr_possible_moves_by_step=[4,1,2,0];

right=1; up=2; left=3; down=4;
directions={'right','up','left','down'};

for a=1:nr_possible_moves_by_step(1)
    action_struct(a).nr=a;
    action_struct(a).direction=directions{a};
    action_struct(a).state=a+1;
    action_struct(a).reward=NaN;
    action_struct(a).done=false;
end

start_state=struct('nr',1,'path',[],'available_actions',...
    1:nr_possible_moves_by_step(1),'is_terminal_state',false,...
    'is_initial_state',true,'location',[0,0],'actions',action_struct);
states_by_step{1}=[start_state]; %each state can be identified by the sequence of moves that led to it
state_nr=1;
states=[states_by_step{1}(:)];
for step=2:(nr_steps+1)
    states_by_step{step}=[];
    for s=1:numel(states_by_step{step-1})
        
        previous_state=states_by_step{step-1}(s);
        
        for m=1:nr_possible_moves_by_step(step-1)
            state_nr=state_nr+1;
            move=previous_state.available_actions(m);
            path=[previous_state.path;move];
            
            clear actions_struct
            
            if numel(path)==1
                available_actions=path(1);
            elseif numel(path)==2
                if ismember(path(end),[left,right])
                    available_actions=[up,down];
                elseif ismember(path(end),[up,down])
                    available_actions=[left,right];
                end
            elseif numel(path)==3
                available_actions=[];
                actions_struct=struct([]);
            end           
            
            x=sum(path==right)-sum(path==left);
            y=sum(path==up)-sum(path==down);
            location_by_state(s,:)=[x,y];            
            
            for a=1:numel(available_actions)
                action=available_actions(a);
                
                switch action
                    case 1
                        direction='right';
                    case 2
                        direction='up';
                    case 3
                        direction='left';
                    case 4
                        direction='down';
                end
                
                actions_struct(a)=struct('direction',direction,...
                    'nr',action,'state',0,'reward',NaN,'done',false);
            end
            
            
            
            state=struct('nr',state_nr,'path',path,'available_actions',...
                available_actions,'is_terminal_state',step==nr_steps,...
                'is_initial_state',false,'location',[x,y],'actions',actions_struct);
            states_by_step{step}=[states_by_step{step}(:);state];
        end
    end
    states=[states(:);states_by_step{step}(:)];
end

states_by_path=containers.Map();

for s=1:numel(states)    
    
    states_by_path(num2str(states(s).path))=states(s);
    
end

nr_states=numel(states);

for s=1:nr_states
    
    state=states(s);
    
    for a=1:numel(state.actions)   
        next_state=states_by_path(num2str([state.path;state.actions(a).nr]));
        states(s).actions(a).state=next_state.nr;        
    end
    
end


%build the transition matrix
nr_moves=max(nr_possible_moves_by_step);
T=zeros(nr_states,nr_states,nr_moves);
for from=1:numel(states)
    from_path=states(from).path;
    for to=1:numel(states)
        to_path=states(to).path;
        for m=1:nr_moves
            test_path=[from_path(:);m];
            if numel(test_path)==numel(to_path)
                T(from,to,m)=all(test_path==to_path);
            else
                T(from,to,m)=0;
            end
        end
    end
end

%% generate reward functions
mean_reward=1; std_reward=2;
r_max=15; r_min=-15;
for m=1:nr_moves
    baseline_rewards(:,:,m)=T(:,:,m).*mvnrnd(mean_reward*ones(nr_states,nr_states),...
        std_reward*repmat(eye(nr_states),[1,1,nr_states]),nr_states);
    
    baseline_rewards(:,:,m)=min(r_max,max(r_min,roundToMultipleOf(baseline_rewards(:,:,m),0.5)));
end

baseline_score=evaluateMDP(T,baseline_rewards,1,3);

baseline_mdp.start_state=1;
baseline_mdp.horizon=3;
baseline_mdp.T=T;
baseline_mdp.rewards=baseline_rewards;
baseline_mdp.states=states;
baseline_mdp.states_by_step=states_by_step;
baseline_mdp.states_by_path=states_by_path;


%% optimize MDP
old_score=baseline_score;

final=false;

reps=80;

%TODO: update the criteria used to filter 
baseline_mdps=repmat(baseline_mdp, [reps,1]);
score_nonmyopic=repmat(baseline_score, [reps,1]);
score_myopic=repmat(baseline_score, [reps,1]);

clear mdps_myopic score_myopic optimality_myopic properties_myopic
clear mdps_nonmyopic score_nonmyopic optimality_nonmyopic properties_nonmyopic
for rep=1:reps
    %myopic_is_optimal=mod(rep,4)==0;
    %myopic_is_optimal=true;
    
    for m=1:nr_moves
        baseline_mdps(rep).rewards(:,:,m)=T(:,:,m).*mvnrnd(mean_reward*ones(nr_states,nr_states),...
            std_reward*repmat(eye(nr_states),[1,1,nr_states]),nr_states);

        baseline_mdps(rep).rewards(:,:,m)=min(r_max,max(r_min,round(100*baseline_rewards(:,:,m))/100));
    end

    [mdps_nonmyopic(rep),score_nonmyopic(rep),optimality_nonmyopic(rep),properties_nonmyopic(rep)]=...
        optimizeMouselabMDP(baseline_mdps(rep),1000,r_max-r_min,false)
    
    [mdps_myopic(rep),score_myopic(rep),optimality_myopic(rep),properties_myopic(rep)]=...
        optimizeMouselabMDP(baseline_mdps(rep),1000,r_max-r_min,true)
    
end

if (isfield(mdps_myopic,'R'))
    mdps_myopic=rmfield(mdps_myopic,'R')
end
if (isfield(mdps_nonmyopic,'R'))
    mdps_nonmyopic=rmfield(mdps_nonmyopic,'R')
end

[sorted_scores_myopic,good_myopic_problems]=sort(optimality_myopic.*([properties_myopic.R_min]>=0),'descend');
[sorted_scores_nonmyopic,good_nonmyopic_problems]=sort(optimality_nonmyopic.*([properties_nonmyopic.R_min]>=0),'descend');

experiment(1:15)=mdps_nonmyopic(good_nonmyopic_problems(1:15))
experiment(16:20)=mdps_myopic(good_myopic_problems(1:5))

properties_experiment(1:15)=properties_nonmyopic(good_nonmyopic_problems(1:15));
properties_experiment(16:20)=properties_myopic(good_myopic_problems(1:5));

R_total_experiment=reshape([properties_experiment.R_total],[4,20])
avg_R_total=mean(R_total_experiment,2)
min_R_total=min(R_total_experiment')

experiment_unscaled=experiment;
properties_unscaled=properties_experiment;

save MDPExperiment_unscaled experiment_unscaled
save properties_unscaled properties_unscaled

save mdps_nonmyopic mdps_nonmyopic
save mdps_myopic mdps_myopic
save properties_myopic properties_myopic
save properties_nonmyopic properties_nonmyopic


%%
final=false;

load MDPExperiment_unscaled 
load properties_unscaled

experiment=experiment_unscaled

min_return=min([properties_unscaled.R_total])

scaling_factor=2;
shift=floorToMultipleOf(scaling_factor*min_return/experiment(1).horizon,scaling_factor*0.5);


%shift rewards so that the mininum return for worst-case performance is 0.
nr_states=size(experiment(1).T,1);
nr_actions=size(experiment(1).T,3);
for e=1:numel(experiment)
    experiment(e).rewards(experiment(e).T(:)>0)=scaling_factor*experiment_unscaled(e).rewards(experiment(e).T(:)>0)-shift;
    properties_experiment(e)=evaluateMouselabMDP(experiment(e).T,...
        experiment(e).rewards,experiment(e).start_state,experiment(e).horizon,...
        e>15);        
    %evaluateMouselabMDP(current_MDP.T,current_MDP.rewards,current_MDP.start_state,current_MDP.horizon,e>15)
end

R_total_experiment=reshape([properties_experiment.R_total],[4,20])
avg_R_total=mean(R_total_experiment,2)
min_R_total=min(R_total_experiment')

std_R_total=std(R_total_experiment,[],2)

all_R=[];
for e=1:numel(experiment)
    all_R=[all_R; experiment(e).rewards(experiment(e).T(:)>0)];
end
sigma_R=std(all_R)
mu_R=mean(all_R)

save MouselabExperimentFeb25 experiment

%% 
clear
FINAL=true;
load mcrl-experiment/data/MouselabExperimentFeb25.mat

for e=1:numel(experiment)
    properties_experiment(e)=evaluateMouselabMDP(experiment(e).T,...
        experiment(e).rewards,experiment(e).start_state,experiment(e).horizon,...
        e>15);
end

trial_properties=properties_experiment;
for t=1:numel(experiment)
    
    for s=1:numel(experiment(t).states)
        state=experiment(t).states(s);
        for a=1:numel(state.actions)
            rewards(state.actions(a).state)=state.actions(a).reward;
        end
    end
    %swapped_states=experiment(t).states;
    %swapped_states(12)=experiment(t).states(13);
    %swapped_states(12).nr=12;
    %swapped_states(13)=experiment(t).states(12);
    %swapped_states(13).nr=13;
    %swapped_states(16)=experiment(t).states(17);
    %swapped_states(16).nr=16;
    %swapped_states(17)=experiment(t).states(16);
    %swapped_states(17).nr=17;
    %experiment(t).states=swapped_states;
    
    trial_properties(t).reward_by_state=rewards;
end

save trial_properties_March1 trial_properties
save experiment_March1 experiment

if FINAL
    save mcrl-experiment/data/trial_properties_March1.mat trial_properties
    save mcrl-experiment/data/experiment_March1.mat experiment
end

%%
states=experiment(1).states;

for e=1:numel(experiment)
    
    experiment(e).actions=1:4;
    
    for s=1:nr_states
        
        state=states(s);
        
        from=state.nr;
        
        for a=1:numel(state.actions)
            to=experiment(e).states(s).actions(a).state;
            action=experiment(e).states(s).actions(a).nr;
            experiment(e).states(s).actions(a).reward=...
                experiment(e).rewards(from,to,action);
            
            experiment(e).states(to).reward=experiment(e).rewards(from,to,action);
            
            experiment(e).states(s).actions(a).done=...
                experiment(e).states(to).is_terminal_state;
        end
        
    end
end

if final
    experiment_json=rmfield(rmfield(experiment,'states_by_step'),'states_by_path');
    savejson('',experiment_json,'MouselabMDPExperimentFeb25.json')
end

for e=1:numel(experiment)
    layout(e,:)=experiment(e).states;
end

if final
    savejson('',layout,'MouselabMDPExperimentLayout.json')
    save MouselabMDPExperimentFeb25.mat experiment
end
%%
load('MouselabExperimentFeb25.mat')

final=false;

for e=1:numel(experiment)
    experiment(e).states_by_path=states_by_path;
    for s=1:numel(experiment(e).states)
        for a=1:numel(experiment(e).actions)
            if ismember(a,experiment(e).states(s).available_actions)
                experiment(e).nextState(s,a)=find(experiment(e).T(s,:,a));
            else
                experiment(e).nextState(s,a)=NaN;
            end
        end
    end
end
save('MouselabExperimentFeb25.mat','experiment')

%{
load('MouselabExperimentFeb25.mat')
layout=loadjson('MouselabMDPExperimentLayout.json')

final=false;

for e=1:numel(experiment)
    
    experiment(e).actions=1:4;
    states=experiment(e).states;
    nr_states=numel(states);
    
    layout{e}{1}.reward=0;
    
    for s=1:nr_states
        
        state=states(s);
        
        from=state.nr;
        
        for a=1:numel(state.actions)
            to=experiment(e).states(s).actions(a).state;
            action=experiment(e).states(s).actions(a).nr;

            layout{e}{to}.reward=experiment(e).rewards(from,to,action);
        end
        
    end
end

if final
    savejson('',layout,'MouselabMDPExperimentLayout.json')
end
%}
%% 
%experiment=loadjson('MouselabMDPExperiment.json')
load MouselabExperimentFeb25.mat
final=false;

for e=1:numel(experiment)
    properties_experiment(e)=evaluateMouselabMDP(experiment(e).T,...
        experiment(e).rewards,experiment(e).start_state,experiment(e).horizon,...
        e>15);
end


for e=1:numel(experiment)
    
    states=experiment(e).states(1);
    nr_states=numel(states);
        
    experiment(e).states(1).reward=0;
    for s=1:nr_states
        
        state=states(s);
        
        from=state.nr;
        
        for a=1:numel(state.actions)
            action=state.actions(a).nr;
            to=state.actions(a).state;
            experiment(e).states(to).reward=experiment(e).rewards(from,to,action);
        end
        
    end
end


if final
    save('MouselabExperimentFeb25.mat','experiment')
    experiment_json=rmfield(experiment,'states_by_path')
    savejson('',experiment_json,'MouselabMDPExperimentFeb25.json')
end

%% create 8 trial version of the experiment
load trial_properties_March1
load experiment_March1

selected_trials=[1:6,16:17];
control_experiment=experiment(selected_trials);
control_trial_properties=trial_properties(selected_trials);
experiment_json=rmfield(control_experiment,'states_by_path')
savejson('',experiment_json,'/Users/Falk/Dropbox/PhD/Metacognitive RL/mcrl-experiment/MouselabMDPControlExperiment.json')
save ControlExperiment control_experiment
save('/Users/Falk/Dropbox/PhD/Metacognitive RL/mcrl-experiment/data/trial_properties_control_experiment.mat', 'control_trial_properties')

%% compute object-level PRs
load ControlExperiment
load('/Users/Falk/Dropbox/PhD/Metacognitive RL/mcrl-experiment/data/trial_properties_control_experiment.mat')

addpath('~/Dropbox/PhD/Gamification/')

for t=1:numel(control_experiment)
    [pseudoreward_matrices{t},V{t},policy{t}]=optimalPseudoRewards(....
        control_experiment(t).T,control_experiment(t).rewards,...
        control_experiment(t).horizon,1,false,true)
    
    for step=1:3
        for from=1:17
            for to=1:17
                pseudorewards{t}{step}{from}{to}=pseudoreward_matrices{t}(from,to,step);
            end
        end
    end
end

savejson('',pseudorewards,'/Users/Falk/Dropbox/PhD/Metacognitive RL/mcrl-experiment/ObjectLevelPRs.json')
savejson('',pseudorewards,'/Users/Falk/Dropbox/PhD/Metacognitive RL/mcrl-experiment/experiment/static/json/ObjectLevelPRs.json')

load trial_properties_March1
load experiment_March1

selected_trials=[1:6,16:17];
control_experiment=experiment(selected_trials);
trial_properties=trial_properties(selected_trials)

save('/Users/Falk/Dropbox/PhD/Metacognitive RL/mcrl-experiment/data/trial_properties_1E.mat','trial_properties')

%% Create trials for retention experiment: 8 training trials and 4 test trials
load trial_properties_March1
load experiment_March1

selected_trials=[1:9,16:18];
retention_experiment=experiment(selected_trials);
retention_trial_properties=trial_properties(selected_trials);
experiment_json=rmfield(retention_experiment,'states_by_path')
savejson('',experiment_json,'/Users/Falk/Dropbox/PhD/Metacognitive RL/mcrl-experiment/RetentionExperiment.json')
save RetentionExperiment retention_experiment
save('/Users/Falk/Dropbox/PhD/Metacognitive RL/mcrl-experiment/data/trial_properties_retention_experiment.mat', 'retention_trial_properties')



%% evaluate the performance of different levels of planning
%{
clear
final=false;

load MouselabExperimentFeb25

for e=1:numel(experiment)
    properties(e)=evaluateMouselabMDP(experiment(e).T,experiment(e).rewards,...
        experiment(e).start_state,experiment(e).horizon,false)
    
    R_max(e)=properties(e).R_max;
    R_min(e)=properties(e).R_min;
end

%max_returns=[5.00,5.00];
for e=1:numel(experiment)

    min_reward=min(properties(e).R_total(1));
    shift=min_reward/(experiment(e).horizon);
    
    experiment(e).getActions = @(state) find(any(squeeze(experiment(e).T(state,:,:))));
    
    max_return=20+randn();

    experiment(e).rewards=(experiment(e).rewards.*experiment(e).T-shift)/(properties(e).R_max-shift)*max_return;
    
    
    for s=1:numel(experiment(e).states)
        
        available_actions=experiment(e).states(s).actions;
        nr_actions=numel(available_actions);
        for a=1:nr_actions            
            
            action=available_actions(a);
            
            next_state=experiment(e).states(s).actions(a).state;
            experiment(e).states(s).actions(a).reward=...
                experiment(e).rewards(experiment(e).states(s).nr,next_state,action.nr);
            
            experiment(e).states(next_state).reward=...
                experiment(e).rewards(experiment(e).states(s).nr,next_state,action.nr);
        end
    end
    
    properties(e)=evaluateMouselabMDP(experiment(e).T,experiment(e).rewards,...
        experiment(e).start_state,experiment(e).horizon,false)
end
average_reward_by_planning_horizon=mean(reshape([properties(1:15).R_total],[4,15]),2)
std_reward_by_planning_horizon=std(reshape([properties(1:15).R_total],[4,15]),[],2)
max_reward_by_planning_horizon=max(reshape([properties.R_total],[4,20])')
min_reward_by_planning_horizon=min(reshape([properties.R_total],[4,20])')

if final
    save('MouselabMDPExperiment_normalized3.mat','experiment')
    
    experiment_json=rmfield(rmfield(experiment,'getActions'),'states_by_path');
    savejson('',experiment_json,'MouselabMDPExperiment3.json');
end
%}

%% 
%{
load('MouselabMDPExperiment_normalized.mat')

for e=1:numel(experiment)
    trial_properties(e)=evaluateMouselabMDP(experiment(e).T,experiment(e).rewards,...
        experiment(e).start_state,experiment(e).horizon,false)    
end
%}
%save trial_properties trial_properties