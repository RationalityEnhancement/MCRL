clear all;

% tmp = matlab.desktop.editor.getActive;
% cd(fileparts(tmp.Filename));

% if strcmp(getenv('USER'),'paulkrueger')
% %     cd ~/Desktop/Tom_Griffiths/collapsingBoundsExp/matlab_code/
%     addpath(genpath('~/Documents/MATLAB/Add-Ons/Toolboxes/Markov Decision Processes (MDP) Toolbox'))
% else
%     % Falk path goes here
%     cd
%     addpath(genpath(''))
% end

%%
addpath('./MatlabTools')
nTrials = 30;
rewardCorrect = 1;
rewardIncorrect = -1;
discount = 1;
costs=logspace(-3,-1,10);
% costs = 0.0001;

for c=1:numel(costs)
    cost = costs(c);
    s = nTrials*(nTrials+1)/2; % size of state space
    S = nan(s+1,2); % the states
    R = nan(s+1,2); % rewards
    P1 = zeros(s+1); % transition matrix for action 1
    P2 = zeros(s+1); % transition matrix for action 2 (goes to terminal state)
    min_trial = nan(s+1,1); % the minimum possible trial number for a given state
    
    s = 0; % state index
    for t = 1:nTrials
        for i = 1:t % t possible states at *begining* of each trial
            S(s+i,:) = [i, t+1-i]; % number of "heads," number of "tails"
            p_correct = [i/(t+1), (t+1-i)/(t+1)];
            R(s+i,:) = p_correct * rewardCorrect + (1-p_correct)*(rewardIncorrect); % reward for guessing each option
            P1(s+i,s+i+t:s+i+t+1) = fliplr(p_correct); % in S, T comes before H
            min_trial(s+i) = t;
        end
        s = s + t;
    end
    
    P2(:,s+1) = 1;
    l = (nTrials-1)*nTrials/2;
    P1(l+1:s+1,s+1) = 1;
    P1 = P1(1:s+1,1:s+1);
    
    P = cat(3,P1,P2);
    R = [-cost*ones(s+1,1), max(R,[],2)]; % zero reward for action 1
    R(s+1,:) = [0,0];
    S(s+1,:) = [-1,-1];
    min_trial(s+1) = 30;
    
    for j = 1:s
        if sum(S(j,:)) == nTrials+1
            R(j,1) = R(j,2) - cost;
        end
    end
    
    
    [values, policy] = mdp_finite_horizon (P, R, discount, nTrials);
    
    lightbulb_mdp(c).v_star=values(:,1);
    lightbulb_mdp(c).pi_star=policy(:,1);
    lightbulb_mdp(c).Q_star=getQFromV(values(:,1),P,R);
    lightbulb_mdp(c).states=S;
    lightbulb_mdp(c).T=P;
    lightbulb_mdp(c).R=R;
    
    
    pseudoR_matrix = get_pseudoreward_matrix(S,values,min_trial,discount,R);
    save('mdp_pseudorewards','values','policy','P','R','S','pseudoR_matrix','discount','nTrials','min_trial')
    save(['mdp_pseudorewardsPlusExpectedR_',num2str(nTrials),'trials'],'values','policy','P','R','S','pseudoR_matrix','discount','nTrials','min_trial')
    save(['mdp_pseudorewards_compareSARSA_',num2str(nTrials),'trials'],'values','policy','P','R','S','pseudoR_matrix','discount','nTrials','min_trial')
    
    % save in format that falk requested
    PR_observe = pseudoR_matrix(:,1:nTrials,1);
    PR_bet = pseudoR_matrix(:,1:nTrials,2);
    save(['TverskyEdwards_pseudorewards_',num2str(nTrials),'trials'],'PR_observe','PR_bet','S')
    
    pseudoR_matrix = pseudoR_matrix(:,1:nTrials,:);
    lightbulb_mdp(c).optimal_PR = pseudoR_matrix; %[PR_observe(:,1),PR_bet(:,1)] + ER;

    lightbulb_mdp(c).cost=cost;
end

save('../results/lightbulb_problem.mat','lightbulb_mdp') 