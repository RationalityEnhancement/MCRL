function [ action ] = noObservationPolicy(state, mdp)

[actions,mdp]=mdp.getActions(state);

%choose the first observation action that has not been taken already
action_chosen=false;
for a=1:numel(actions)
    if actions(a).is_decision_mechanism
        action = actions(a);
        return 
    end
end

end