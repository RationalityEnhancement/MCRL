%% Set Variables

setenv('DYLD_LIBRARY_PATH', '/usr/local/bin/');

addpath('./GeneralMDP/')
addpath('./MatlabTools')

rewardCorrect = 1;
constantArm = -1;
rewardIncorrect = 0;
discount = 1;

% nr_arms= 3;
% nr_balls = 6;
% cost = 0.01;
%% Call Python Script 

commandStr = ['python ./GeneralMDP/generate_state_matrices.py ',...
    int2str(nr_balls),' ',int2str(nr_arms),' ',num2str(rewardCorrect),...
    ' ',num2str(cost),' ',num2str(constantArm)];
[status, commandOut] = system(commandStr);

%% Load and Solve MDP

load('./GeneralMDP/file')

nr_states = size(states,1);
nr_arms = size(states,2)/2;
% nTrials = sum(states(end-1,:))-2*nr_arms;

[values, policy] = mdp_finite_horizon (transition, rewards, discount, nr_balls+1);
Q_star=getQFromV(values(:,1),transition,rewards);

%% Check if Early Terminated

steps = nr_arms*2;
early_terminated = 0;
all_terminated = 1;
for i=1:nr_states
    if sum(states(i,:)) == steps
        if all_terminated && not(policy(i,1) == nr_arms+1)
            all_terminated = 0;
            steps = steps+1; %skip until next level
        else
            all_terminated = 1;
        end
    elseif all_terminated
        early_terminated = 1;
        disp(num2str(steps-2*nr_arms));
        break
    end
end
    
%% Calculate PRs (without rewards)

exact_PR = nan(nr_states,nr_arms+1);
for s=1:nr_states
    for a=1:nr_arms+1
        next_s = find(not(transition(s,:,a) == 0));
        evp = 0;
        for isp=1:numel(next_s)
            sp = next_s(isp);
            evp = evp + transition(s,sp,a)*values(sp,1);
        end
        exact_PR(s,a) = evp-values(s,1);
    end
end

%     exact_PR = nan(nr_states,nr_arms+1);
%     exact_PR_Q = nan(nr_states,nr_arms+1);
%     for s=1:nr_states
%         for a=1:nr_arms+1
%             exact_PR_Q(s,a) = Q_star(s,a) - values(s,1);
%             next_s = find(not(transition(s,:,a) == 0));
%             evp = 0;
%             for isp=1:numel(next_s)
%                 sp = next_s(isp);
%                 evp = evp + transition(s,sp,a)*values(sp,1);
%             end
%             exact_PR(s,a) = evp+rewards(s,a)-values(s,1);
%         end
%     end

%% Save

nlightbulb_mdp.cost=cost;
nlightbulb_mdp.nr_arms=nr_arms;
nlightbulb_mdp.nr_balls=nr_balls;

nlightbulb_mdp.states=states;
nlightbulb_mdp.T=transition;
nlightbulb_mdp.R=rewards;

nlightbulb_mdp.v_star=values(:,1);
nlightbulb_mdp.pi_star=policy(:,1);
nlightbulb_mdp.Q_star=Q_star;
nlightbulb_mdp.exact_PR=exact_PR;

save(['../results/', num2str(nr_arms),'lightbulb_problem.mat'],'nlightbulb_mdp','-v7.3') 