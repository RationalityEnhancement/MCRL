function [m, s2] = gp_call(hyp, meanfunc, covfunc, likfunc, X_sample, F_sample, xxx)

hyp2 = hyp;
error = 1;
while error == 1
    error = 0;
    try
       [m, s2] = gp(hyp2, @infExact, meanfunc, covfunc, likfunc, X_sample, F_sample, xxx);
    catch
       error = 1;
       if hyp2.lik == -inf, hyp2.lik = - 9; end
       hyp2.lik = hyp2.lik + 1;
    end
end

end