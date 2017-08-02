function computation = satisficingPolicy(state,mdp)

[computations,mdp]=mdp.getActions(state);

observed_outcomes = state.observations(not(isnan(state.observations)));


end