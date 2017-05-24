function nv=nvoc(nTrials,init,cost)
rewardCorrect = 1;
rewardIncorrect = 0;
discount = 1;
s = nTrials*(nTrials+1)/2; % size of state space
S = nan(s+1,2); % the states
R = nan(s+1,2); % rewards
P1 = zeros(s+1); % transition matrix for action 1
P2 = zeros(s+1); % transition matrix for action 2 (state doesn't change)
min_trial = nan(s+1,1); % the minimum possible trial number for a given state

s = 0; % state index
for t = 1:nTrials
    for i = 1:t % t possible states at *begining* of each trial
        a = [init(1)+i-1, init(2)+t-i];
        S(s+i,:) = a; % number of "heads," number of "tails"
        p_correct = [a(1)/sum(a), a(2)/sum(a)];
        R(s+i,:) = p_correct * rewardCorrect +(1-p_correct)*(rewardIncorrect); % reward for guessing each option   
        P1(s+i,s+i+t:s+i+t+1) = fliplr(p_correct);
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


[values, policy] = mdp_finite_horizon (P, R, discount, nTrials);


i1 = [init(1)+1,init(2)];
tic,f1=find(all(repmat(i1,size(S,1),1)==S,2));

i2 = [init(1),init(2)+1];
tic,f2=find(all(repmat(i2,size(S,1),1)==S,2));

nv = (init(1)*values(f1,1) + init(2)*values(f2,1) - max(init))/sum(init) - cost;
end