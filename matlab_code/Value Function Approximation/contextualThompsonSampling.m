function [ action ] = contextualThompsonSampling(state, mdp, glm,actionFeatures)

%Thompson sampling for contextual bandit problems
w_hat=glm.sampleCoefficients();

[actions,mdp]=mdp.getActions(state);
for a=1:numel(actions) %used to be parfor
    action_features=actionFeatures(state,actions(a),mdp);
    Q_hat(a)=dot(w_hat(mdp.action_features),action_features);
end
%a_max=argmaxWithRandomTieBreak(Q_hat);
a_max=argmax(Q_hat);
action=actions(a_max);

end