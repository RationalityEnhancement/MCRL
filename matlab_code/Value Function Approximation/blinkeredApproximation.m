function [VOC_blinkered,V_blinkered,Q_blinkered,pi_blinkered]=blinkeredApproximation(T,R,h,c,bot,cost)
%blinkered approximation the value function of a meta-level MDP as proposed
%by Hay et al. (2012).

%T(from,to,c) is the transition probability
%R(from,to,c)is the reward function
%c is the meta-level action to be evaluated
%bot is the meta-level action that terminates deliberation and takes action
%h is an upper bound on the number of computations

T_blinkered=T(:,:,[c,bot]);
R_blinkered=R(:,:,[c,bot]);

[V_blinkered, pi_blinkered] = mdp_finite_horizon (T_blinkered, R_blinkered, 1, h);

Q_blinkered = getQFromV(V_blinkered,T_blinkered,R_blinkered,1);

VOC_blinkered=T(:,:,c)*V_blinkered-cost-R(:,end,bot);

end