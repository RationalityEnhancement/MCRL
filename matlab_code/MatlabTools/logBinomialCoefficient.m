function log_binomial_coefficient  = logBinomialCoefficient(n,k)
%computes the logarithm of the binomial coefficient for n-choose-k.

for i=1:numel(k)
    log_binomial_coefficient(i,1)=sum(log(1:n))-sum(log(1:(n-k(i))))-sum(log(1:k(i)));
end

log_binomial_coefficient=reshape(log_binomial_coefficient,size(k));

end

