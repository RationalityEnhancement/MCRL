function p_y=posteriorPredictiveLogisticRegressionVB(x,mu_prior,Pi_prior)
%Jaakola, & Jordan (1996). A variational approach to Bayesian logistic
%regression models and their extensions.

putative_y=1;
[mu_posterior,Pi_posterior,ksi]=logisticRegressionVB(putative_y,x,mu_prior,Pi_prior);

lambda=@(ksi) (1/2-sigmoid(ksi))/(2*ksi);


%p_y=sigmoid(ksi)/exp(lambda(ksi)*ksi^3/2)*exp(1/2*mu_posterior'*Pi_posterior*mu_posterior-...
%    1/2*mu_prior'*Pi_prior*mu_prior)*sqrt(det(Pi_prior)/det(Pi_posterior));

%Equation 10
log_p_y=log(sigmoid(ksi))-ksi/2-lambda(ksi)*ksi^2-1/2*mu_prior'*Pi_prior*mu_prior+...
    1/2*mu_posterior'*Pi_posterior*mu_posterior+1/2*(logdet(Pi_prior)-logdet(Pi_posterior));

p_y=exp(log_p_y);

end