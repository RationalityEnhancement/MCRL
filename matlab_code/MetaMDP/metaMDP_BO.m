for c=1:numel(costs)
    [x, fx, X_sample, F_sample, result] = policySearchMetaMDP(c);
    cd('../../')
    w_BO = x;
    extracBOpolicy;
end