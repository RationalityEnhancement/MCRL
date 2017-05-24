function VPI=valueOfPerfectInformationMultiArmBernoulli(alphas,betas,c)
%alphas: vector of alpha parameters of the Bernoulli distributions over the
%arms' reward probabilities
%betas: vector of the beta parameters of the Bernoulli distribtions over
%the arms' reward probabilities
%c: index of the action about which perfect information is being obtained

E_correct=alphas(:)./(alphas(:)+betas(:));
[sorted_values,arm_numbers]=sort(E_correct,'descend');

mu_alpha=sorted_values(1);
mu_beta=sorted_values(2);
alpha=arm_numbers(1);
beta=arm_numbers(2);


if c==alpha %information can only be valuable by revealing that alpha is actually suboptimal
    lb=0;
    ub=mu_beta;
    VPI = integral(@(x) betapdf(x,alphas(c),betas(c)).*(mu_beta-x),lb,ub);    
else
    %information can only be valuable by revealing that c is actually
    %better than alpha
    lb=mu_alpha;
    ub=1;
    VPI = integral(@(x) betapdf(x,alphas(c),betas(c)).*(x-mu_alpha),lb,ub);        
end

end