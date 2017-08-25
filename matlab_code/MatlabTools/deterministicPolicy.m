function [ action ] = deterministicPolicy(state, mdp, weights,use_fast_VOC1_approximation)

if not(exist('use_fast_VOC1_approximation','var'))
    use_fast_VOC1_approximation=true;
end

%Thompson sampling for contextual bandit problems
w_hat=weights;

[actions,mdp]=mdp.getActions(state);
for a=1:numel(actions) %parfor
    action_features=mdp.extractActionFeatures(state,actions(a),use_fast_VOC1_approximation);
    Q_hat(a)=dot(w_hat(mdp.action_features),action_features);
end
%a_max=argmaxWithRandomTieBreak(Q_hat);
a_max=argmax(Q_hat);
action=actions(a_max);

end