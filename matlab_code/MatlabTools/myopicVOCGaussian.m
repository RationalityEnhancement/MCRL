function VOC=myopicVOCGaussian(mu,sigma,c,cost,sigma_prior)
%mu: vector of expected values of the returns of all possible actions
%sigma: corresponding standard deviations
%c: index of the action about which perfect information is being obtained

if not(exist('sigma_prior','var'))
    sigma_prior=sigma;
end

[mu_sorted,pos_sorted]=sort(mu,'descend');

mu_alpha=mu_sorted(1);
alpha=pos_sorted(1);

mu_beta=mu_sorted(2);
beta=pos_sorted(2);

if c==alpha
    %information is valuable if it reveals that action c is suboptimal
    
    %To change the decision, the sampled value would have to be less than
    %ub
    tau_total=1/sigma(c)^2+1/sigma_prior^2;
    ub=sigma(c)^2*(tau_total*mu_beta-1/sigma_prior(c)^2*mu_alpha);
    
    VOC=sigma(c)^2/2*normpdf(ub,sigma(c))-...
        (mu_alpha-mu_beta)*normcdf(ub,mu_alpha,sigma(c))-cost;

else
    %information is valuable if it reveals that action is optimal    
    
    %To change the decision, the sampled value would have to be larger than
    %lb
    tau_total=1/sigma(c)^2+1/sigma_prior^2;
    lb=sigma(c)^2*(tau_total*mu_alpha-1/sigma_prior(c)^2*mu(c));
    
    
    VOC=sigma(c)^2/2*normpdf(lb,mu(c),sigma(c))-...
        (mu_alpha-mu(c))*(1-normcdf(lb,mu(c),sigma(c)))-cost;
end

end