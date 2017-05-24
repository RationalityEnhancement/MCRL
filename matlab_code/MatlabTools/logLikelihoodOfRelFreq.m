function [ ll ] = logLikelihoodOfRelFreq( theta_hat,N, theta)
%logLikelihoodOfRelFreq( theta_hat,n, theta) returns the log-liklehood of
%observing a relative frequency of theta_hat in n outcomes drawn from
%Bernoulli(theta).

ll=0;
for i=1:numel(theta_hat)
    n=round(theta_hat(i)*N(i)); %number of observations in which the event occurred
    k=round((1-theta_hat(i))*N(i)); %number of observations in which the event did not occur

    ll=ll+n.*log(theta(i))+k.*log(1-theta(i))+logBinomialCoefficient(N(i),n);
end

ll=max(sum(N(:))*log(eps),ll);

end