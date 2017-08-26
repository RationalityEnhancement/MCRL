function VPI=VPI_all_MultiArmBernoulli(alphas,betas)
%alphas: vector of alpha parameters of the Bernoulli distributions over the
%arms' reward probabilities
%betas: vector of the beta parameters of the Bernoulli distribtions over
%the arms' reward probabilities
%c: index of the action about which perfect information is being obtained

%full information would allow you to pick the harm for which the
%probability of winning is highest and its return would be max_a p_a
ER_full_information = EVMaxBeta(alphas, betas);

EVs =alphas./(alphas+betas);
ER_current_information = max(EVs);

VPI = ER_full_information - ER_current_information;

end