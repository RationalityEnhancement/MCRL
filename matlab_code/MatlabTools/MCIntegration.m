function [density_estimate,sem]=MCIntegration(density_function,values,sampled_parameters)
%Given samples from P(alpha_i,beta_i|O=o), estimate p(theta_i|O=o).
%density
%density=density_function(values,parameters) returns the a n x k
%probability density matrix where each row corresponds to one entry of values
%and each column corresponds to one value of parameters.
%values: n x m vector of n m-dimensional points whose densities shall be estimated
%sampled_parameters: k x l matrix containing k vectors of l parameters

density_matrix=density_function(values,sampled_parameters');
density_estimate=mean(density_matrix,1)';
sem=std(density_matrix,1)'/sqrt(size(density_matrix,2));
    
end