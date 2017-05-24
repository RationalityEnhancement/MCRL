function VOC1=VOC1MultiArmBernoulli(alphas,betas,c,cost)
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

p_reward=alphas(c)/(alphas(c)+betas(c));

if c==alpha %information can only be valuable by revealing that alpha is actually suboptimal
        
        EU_after=p_reward*(alphas(c)+1)/(alphas(c)+betas(c)+1)+...
            (1-p_reward)*max(mu_beta,alphas(c)/(alphas(c)+betas(c)+1));        

else
    %information can only be valuable by revealing that c is actually
    %better than alpha
    
    EU_after=p_reward*max(mu_alpha,alphas(c)+1)/(alphas(c)+betas(c)+1)+...
        (1-p_reward)*mu_alpha;
end

delta_EU=EU_after - mu_alpha;

VOC1=delta_EU-cost;

end