function [ action ] = deterministicPolicy(state, mdp, weights)

%Thompson sampling for contextual bandit problems
w_hat=weights;

[actions,mdp]=mdp.getActions(state);
for a=1:numel(actions) %parfor
    action_features=mdp.extractActionFeatures(state,actions(a));
    Q_hat(a)=dot(w_hat(mdp.action_features),action_features);
end
%a_max=argmaxWithRandomTieBreak(Q_hat);
a_max=argmax(Q_hat);
action=actions(a_max);

end