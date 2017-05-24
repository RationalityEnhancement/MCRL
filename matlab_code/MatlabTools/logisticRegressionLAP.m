function [mu_posterior,Pi_posterior,log_model_evidence]=logisticRegressionLAP(ys,xs,mu_prior,Pi_prior)
%Laplace approximation to Bayesian logistic regression
log_likelihood=@(w) sum( ys.*log(sigmoid(xs*w) ))+ sum( (1-ys).*log(1-sigmoid(xs*w)));
log_prior=@(w) -1/2*log(2*pi)+1/2*logdet(Pi_prior)-1/2*(w-mu_prior)'*Pi_prior*(w-mu_prior);
log_joint=@(w) log_likelihood(w)+log_prior(w);
gradient=@(w) sum(repmat((ys-sigmoid(xs*w)),[1,nr_regressors]).*xs)'-Pi_prior*(w-mu_prior);
hessian=@(w) -(xs.*repmat(sigmoid(xs*w),[1,nr_regressors]))'*(xs.*repmat(1-sigmoid(xs*w),[1,nr_regressors]))-Pi_prior;

nr_regressors=size(xs,2);

w0=mu_prior;
options=optimset('GradObj','on','Hessian','on','Display','off');
[mu_posterior,~,flag,~,gradient,H]=fminunc(@(w) objective_function(w,xs,ys,mu_prior,Pi_prior), w0, options);
Pi_posterior=H;

log_model_evidence=log_joint(mu_posterior)+nr_regressors/2*log(2*pi)-1/2*logdet(Pi_posterior);

end

function [val,grad,hess]=objective_function(w,xs,ys,mu_prior,Pi_prior)
log_likelihood=@(w) sum( ys.*log(sigmoid(xs*w) ))+ sum( (1-ys).*log(1-sigmoid(xs*w)));
log_prior=@(w) -1/2*log(2*pi)+1/2*logdet(Pi_prior)-1/2*(w-mu_prior)'*Pi_prior*(w-mu_prior);
log_joint=@(w) log_likelihood(w)+log_prior(w);
nr_regressors=size(xs,2);
gradient=@(w) sum(repmat((ys-sigmoid(xs*w)),[1,nr_regressors]).*xs)'-Pi_prior*(w-mu_prior);
hessian=@(w) -(xs.*repmat(sigmoid(xs*w),[1,nr_regressors]))'*(xs.*repmat(1-sigmoid(xs*w),[1,nr_regressors]))-Pi_prior;

val=-log_joint(w);
grad=-gradient(w);
hess=-hessian(w);
end

