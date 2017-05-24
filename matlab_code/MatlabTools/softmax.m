function p=softmax(values,beta)
    if not(exist('beta','var'))
        beta=1;
    end
    p=exp(beta*values)/sum(exp(beta*values));
end