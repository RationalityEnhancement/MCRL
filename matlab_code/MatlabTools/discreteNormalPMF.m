function [pmf,values]=discreteNormalPMF(values,mu,sigma)
    %pmf is the probability mass that the discretized, truncated normal
    %distribution over values assigns to the interval centered on each of
    %the values.
    
    delta=mean(diff(values));
    lower_bounds=values-delta/2;
    upper_bounds=values+delta/2;
    
    cdf_below=normcdf(lower_bounds,mu,sigma);
    cdf_above=normcdf(upper_bounds,mu,sigma);
    
    delta_cdf=cdf_above-cdf_below;
    pmf=delta_cdf/sum(delta_cdf);

end