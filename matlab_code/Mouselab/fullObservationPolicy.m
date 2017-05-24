function [ action ] = fullObservationPolicy(state, mdp)

[actions,mdp]=mdp.getActions(state);

%choose the first observation action that has not been taken already
action_chosen=false;
for a=1:numel(actions)
    if actions(a).is_computation
        if isnan(state.observations(actions(a).state))
            action=actions(a);
            action_chosen=true;
            return
        end
    end
end

if not(action_chosen)
    %if no more computations are available, then stop deliberating and take
    %action
    decision_actions=find([actions.is_decision_mechanism]);
    action=actions(drawSample(decision_actions));    
end

end