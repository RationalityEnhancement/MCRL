function VPI=valueOfPerfectInformation(mu,sigma,c)
%mu: vector of expected values of the returns of all possible actions
%sigma: corresponding standard deviations
%c: index of the action about which perfect information is being obtained

[mu_sorted,pos_sorted]=sort(mu,'descend');

max_val=mu_sorted(1);
max_pos=pos_sorted(1);

secondbest_val=mu_sorted(2);
secondbest_pos=pos_sorted(2);

if c==max_pos
    %information is valuable if it reveals that action c is suboptimal
    ub=secondbest_val;
    lb=mu(c)-3*sigma(c);
    
    %VPI = integral(@(x) normpdf(x,mu(c),sigma(c)).*(secondbest_val-x),lb,ub,'AbsTol',0.01,'RelTol',0.01);    
    VPI = (secondbest_val-mu(c))*(normcdf(ub,mu(c),sigma(c))-normcdf(lb,mu(c),sigma(c)))+...
        sigma(c)^2*(normpdf(ub,mu(c),sigma(c))-normpdf(lb,mu(c),sigma(c)));
    %todo: replace numerical integration by the analytic solution
else
    %information is valuable if it reveals that action is optimal
    ub=mu(c)+3*sigma(c);
    lb=max_val;
    
    if ub>lb
        %VPI = integral(@(x) normpdf(x,mu(c),sigma(c)).*(x-max_val),lb,ub,'AbsTol',0.01,'RelTol',0.01);
        %a=(lb-mu(c))/sigma(c);
        %b=(ub-mu(c))/sigma(c);
        VPI = (mu(c)-max_val)*(normcdf(ub,mu(c),sigma(c))-normcdf(lb,mu(c),sigma(c)))-...
            sigma(c)^2*(normpdf(ub,mu(c),sigma(c))-normpdf(lb,mu(c),sigma(c)));

    else
        VPI=0;
    end
end

end