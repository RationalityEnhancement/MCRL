addpath('./MatlabTools/')
addpath('../')

clear all

nr_arms= 5;
nr_balls = 6;
cost = 0.01;

% exact

load(['../results/', num2str(nr_arms),'lightbulb_fit.mat'])

states = nlightbulb_problem.mdp.states;
nr_states=size(states,1)-1;
S=states(1:nr_states,:);
nr_arms = size(states(1,:),2)/2;

valid_states = [1:nr_states]';
s_vstates = size(valid_states);
n_vstates = s_vstates(1);
%% Fill in the VOC1

Q_star=nlightbulb_problem.mdp.Q_star;
pi_meta = zeros(nr_states,1);
voc1 = zeros(nr_states,nr_arms);


for k=1:numel(valid_states)
    i = valid_states(k);
    st = S(i,:);  
    st_m = reshape(st,2,nr_arms)';
    for j=1:nr_arms
        voc1(i,j) = VOC1MultiArmBernoulli(st_m(:,1),st_m(:,2),j,cost);
    end
    voc1(i,nr_arms+1) = 0;
end
voc1(nr_states+1,:) = zeros(nr_arms+1,1);

[m, pi_meta] = max(voc1,[],2);

%% Save

nlightbulb_problem.fit.pi_meta=pi_meta;

disp(['../results/', num2str(nr_arms),'lightbulb_fit.mat'])
save(['../results/', num2str(nr_arms),'lightbulb_fit.mat'],'nlightbulb_problem','-v7.3')