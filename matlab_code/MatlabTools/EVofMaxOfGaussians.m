function [E_max,STD_max]=EVofMaxOfGaussians(mu,sigma)
%[E_max,STD_max]=EVofMaxOfGaussians(mu,sigma,n,c) computes 
%E[max {X_1,...,X_n}] and STD[max {X_1,...,X_n}] for n normally distributed
%random variables X_1,...,X_n with means mu(1),...,mu(n), and standard
%deviations sigma(1),...,sigma(n).

if all(sigma==0)
    E_max=max(mu);
    STD_max=0;
elseif all(sigma>0)
    x_min=min(mu)-5*max(sigma);
    x_max=max(mu)+5*max(sigma);
    E_max=integral( @(x) x.*pdfOfMaxOfGaussians(x,mu,sigma),x_min,x_max,'AbsTol',1e-2);
    STD_max=sqrt( integral( @(x) (x-E_max).^2.*pdfOfMaxOfGaussians(x,mu,sigma),x_min,x_max,'AbsTol',1e-2));
else   %some outcomes are known and some are unknown 
    max_for_sure=max(mu(sigma==0));
    
    mu_uncertain=mu(sigma>0);
    sigma_uncertain=sigma(sigma>0);
    
    x_min=min(mu)-5*max(sigma);
    x_max=max(mu)+5*max(sigma);
    
    E_max=integral( @(x) max(max_for_sure,x).*pdfOfMaxOfGaussians(x,mu_uncertain,sigma_uncertain),x_min,x_max,'AbsTol',1e-2);
    STD_max=sqrt( integral( @(x) (max(max_for_sure,x)-E_max).^2.*pdfOfMaxOfGaussians(x,mu_uncertain,sigma_uncertain),x_min,x_max,'AbsTol',1e-2));
end

end