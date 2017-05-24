function [E_max,STD_max]=EVofMaxIIDGaussians(mu,sigma,n,c)
%[E_max,STD_max]=EVofMaxIIDGaussians(mu,sigma,n,c) computes 
%E[max {c,max {X_1,...,X_n}}] and STD[max {c,max {X_1,...,X_n}}].
%c is optional and its default value is -infinity.

if not(exist('c','var'))
    E_max=mu+sigma*integral( @(x) x.*n.*normpdf(x).*normcdf(x).^(n-1),-inf,inf);
    STD_max=sigma*sqrt( integral( @(x) x.^2.*n.*normpdf(x).*normcdf(x).^(n-1),-inf,inf));
else
    %E[max {c,max {X_1,...,X_n}}]
    E_max=integral( @(x) max(c,x).*n.*normpdf((x-mu)/sigma).*normcdf(x).^(n-1),-inf,inf);
    
    %STD[max {c,max {X_1,...,X_n}}]
    E_c=integral(@(x) max(x,c).*normpdf(x),-inf,inf);
    STD_max=sigma*sqrt( integral( @(x) (max(c,x)-E_c).^2.*n.*normpdf(x).*normcdf(x).^(n-1),-inf,inf));    
end


end