function [ action ] = contextualThompsonSampling(state, mdp, glm)

%Thompson sampling for contextual bandit problems
w_hat=glm.sampleCoefficients();

[actions,mdp]=mdp.getActions(state);
for a=1:numel(actions) %parfor made it extremely slow for some reason
    action_features=mdp.extractActionFeatures(state,actions(a));
    Q_hat(a)=dot(w_hat(mdp.action_features),action_features);
end
%a_max=argmaxWithRandomTieBreak(Q_hat);
a_max=argmax(Q_hat);
action=actions(a_max);

end