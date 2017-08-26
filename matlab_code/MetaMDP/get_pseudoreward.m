function [pseudoR, idx] = get_pseudoreward(state,trial,S,pseudoR_matrix)
% [pseudoR, idx] = get_pseudoreward(state,trial,S,pseudoR_matrix)
% 
% this function is used to get the pseudoreward for a given state-trial
% combination (it could be used during an actual experiment to reference
% the pseudoreward for each trial, given the state)
% 
% state: 1x2 vector where state(1) is number of heads and state(2) is
% number of tails
% 
% trial: scalar number
% 
% S: s x 2 matrix showing the number of heads (coloumn 1) and the number of
% tails (column 2) in each state (rows), where s=total number of states
% 
% pseudoR_matrix: s x (n+1) matrix of pseudorewards for each state and min_trial

idx = find(S(:,1)==state(1) & S(:,2)==state(2));
pseudoR = pseudoR_matrix(idx,trial);