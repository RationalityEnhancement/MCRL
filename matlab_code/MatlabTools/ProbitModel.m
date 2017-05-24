classdef ProbitModel
    
    properties
    end
    
    methods (Static)
        function Y=generateData(X,beta)
            
            if nargin==1
                beta=ProbitModel.sampleBeta(zeros(1,size(X,2)),diag(ones(1,size(X,2))));
            end
            
            p_y=normcdf(X*beta');
            Y=rand(numel(p_y),1)<=p_y;
        end
        
        function beta=sampleBeta(mu,sigma)
            beta=mvnrnd(mu,sigma*diag(ones(size(mu))));
        end
        
        function likelihood=Likelihood(y,X,betas)
            likelihood=prod(normcdf(X(y==1,:)*betas'))*prod(1-normcdf(X(y==0,:))*betas');
        end
        
        function log_likelihood=logLikelihood(y,X,betas)
            log_likelihood=sum(log(max(eps,normcdf(X(y==1,:)*betas'))))+sum(log(max(eps,1-normcdf(X(y==0,:)*betas'))));
        end
        
        function p_betas=prior(betas,sigma_p)
            p_betas=prod(normpdf(betas'/sigma_p),1);
        end
        
        function log_p_betas=logPrior(betas,sigma_p)
            log_p_betas=sum(log(max(eps,normpdf(betas'/sigma_p))),1);
        end
        
        function beta_hat=estimateParameters(y,X,sigma_p)
            nr_parameters=size(X,2);
            
            if exist('sigma_p','var') %MAP
                neg_log_joint=@(betas) -ProbitModel.logPrior(betas,sigma_p)-...
                    ProbitModel.logLikelihood(y,X,betas);
                
                beta_hat=fminunc(neg_log_joint,randn(1,nr_parameters));            
            else %MLE
                neg_log_likelihood=@(betas) -ProbitModel.logLikelihood(y,X,betas);
                
                beta_hat=fminunc(neg_log_likelihood,randn(1,nr_parameters));            
                
            end
        end
        
        function p=predict(X,beta)
            
            p=normcdf(X*beta');
        end
        
        function [BIC,BIC_approximation_to_log_model_evidence,fit]=BIC(y,X)
            MLE=ProbitModel.estimateParameters(y,X);
            fit=ProbitModel.logLikelihood(y,X,MLE);
            nr_parameters=size(X,2);
            nr_datapoints=size(y,1);
            complexity=nr_parameters*log(nr_datapoints);
            BIC=-2*fit+complexity;
            BIC_approximation_to_log_model_evidence=fit-complexity/2;
        end
        
        function [log_model_evidence]=logModelEvidenceNumericalInt(y,X,sigma_p)
            
            nr_dimensions=size(X,2);
            
            beta_min=-10*sigma_p; beta_max=10*sigma_p;
            
            if nr_dimensions==1
            joint_density=@(betas) ProbitModel.prior(betas,sigma_p)+...
                    ProbitModel.likelihood(y,X,betas);

                log_model_evidence=log(integral(joint_density,beta_min,beta_max));
            elseif nr_dimensions==2
                joint_density=@(beta1,beta2) ProbitModel.Prior([beta1,beta2],sigma_p)+...
                    ProbitModel.Likelihood(y,X,[beta1,beta2]);

                log_model_evidence=log(integral2(joint_density,beta_min,beta_max,beta_min,beta_max));
            elseif nr_dimensions==3
                joint_density=@(beta1,beta2,beta3) ProbitModel.Prior([beta1,beta2,beta3],sigma_p)+...
                    ProbitModel.Likelihood(y,X,[beta1,beta2,beta3]);
                
                log_model_evidence=log(integral3(joint_density,beta_min,beta_max,beta_min,beta_max,beta_min,beta_max));
            else
                log_model_evidence=NaN;
            end
            
        end
        
        function [log_model_evidence,posterior_mean,posterior_var,samples]=approximateLogModelEvidence(y,X,sigma_p,nr_samples)
            
            prior.log_density=@(betas) ProbitModel.logPrior(betas,sigma_p);
            likelihood.log_density=@(data,betas) ProbitModel.logLikelihood(data,X,betas);
            likelihood.nr_parameters=size(X,2);
            
            proposal.log_density=@(from,to,sigma) sum(log(max(eps,normpdf((repmat(to,[max(size(to,1),size(from,1))/size(to,1),1])-repmat(from,[max(size(to,1),size(from,1))/size(from,1),1]))/sigma)))',1);
            proposal.propose=@(from,sigma) mvnrnd(from,sigma^2*diag(ones(length(from),1)));
            proposal.sample=@(from,sigma,nr_samples) mvnrnd(from,sigma^2*diag(ones(length(from),1)),nr_samples);
            proposal.initial_value=6./range(X);
            proposal.params=min(proposal.initial_value/10);
            
            [log_model_evidence,~,samples]=...
            lmeChibJeliazkov01(y,prior,likelihood,proposal,nr_samples);
            
            posterior_mean=mean(samples); posterior_var=var(samples);
        end
    end
end