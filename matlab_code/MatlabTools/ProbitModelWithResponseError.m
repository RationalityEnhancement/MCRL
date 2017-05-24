classdef ProbitModelWithResponseError
    
    properties
    end
    
    methods (Static)
        function Y=generateData(X,beta,epsilon)
            
            if nargin==1
                beta=ProbitModelWithResponseError.sampleBeta(zeros(1,size(X,2)),diag(ones(1,size(X,2))));
            end
            
            p_y=(1-epsilon)*normcdf(X*beta')+epsilon*0.5*ones(size(X,1),1);
            Y=rand(numel(p_y),1)<=p_y;
        end
        
        function beta=sampleBeta(mu,sigma)
            beta=mvnrnd(mu,sigma*diag(ones(size(mu))));
        end
        
        function likelihood=likelihood(y,X,betas,epsilon)
            likelihood=prod(epsilon*0.5*ones(sum(y==1),size(betas,1))+(1-epsilon)*normcdf(X(y==1,:)*betas')).*prod(epsilon*ones(sum(y==0),size(betas,1))+(1-epsilon)*(1-normcdf(X(y==0,:))*betas'));
        end
        
        function log_likelihood=logLikelihood(y,X,betas,epsilon)
            log_likelihood=sum(log(max(eps,epsilon*0.5*ones(sum(y==1),size(betas,1))+(1-epsilon)*normcdf(X(y==1,:)*betas'))))+sum(log(max(eps,epsilon*0.5*ones(sum(y==0),size(betas,1))+(1-epsilon)*(1-normcdf(X(y==0,:)*betas')))));
        end
        
        function p_betas=prior(betas,sigma_p)
            p_betas=prod(normpdf(betas'/sigma_p),1);
        end
        
        function log_p_betas=logPrior(betas,sigma_p)
            log_p_betas=sum(log(max(eps,normpdf(betas'/sigma_p))),1);
        end
        
        function beta_hat=estimateParameters(y,X,epsilon,sigma_p)
            nr_parameters=size(X,2);
            
            if exist('sigma_p','var') %MAP
                neg_log_joint=@(betas) -ProbitModelWithResponseError.logPrior(betas,sigma_p)-...
                    ProbitModelWithResponseError.logLikelihood(y,X,betas,epsilon);
                
                beta_hat=fminunc(neg_log_joint,randn(1,nr_parameters));            
            else %MLE
                neg_log_likelihood=@(betas) -ProbitModelWithResponseError.logLikelihood(y,X,betas,epsilon);
                
                beta_hat=fminunc(neg_log_likelihood,randn(1,nr_parameters));            
                
            end
        end
        
        function p=predict(X,beta,epsilon)
            
            p=(1-epsilon)*normcdf(X*beta')+epsilon*0.5*ones(size(X,1),1);
        end
        
        function [BIC,BIC_approximation_to_log_model_evidence,fit]=BIC(y,X,epsilon)
            MLE=ProbitModelWithResponseError.estimateParameters(y,X,epsilon);
            fit=ProbitModelWithResponseError.logLikelihood(y,X,MLE,epsilon);
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
                joint_density=@(beta) ProbitModelWithResponseError.prior(beta,sigma_p)+...
                    ProbitModelWithResponseError.likelihood(y,X,beta);
                log_model_evidence=log(integral(joint_density,beta_min,beta_max));
            elseif nr_dimensions==2
                   joint_density=@(beta1,beta2) ProbitModelWithResponseError.prior([beta1,beta2],sigma_p)+...
                    ProbitModelWithResponseError.likelihood(y,X,[beta1,beta2]);
                
                log_model_evidence=log(integral2(joint_density,beta_min,beta_max,beta_min,beta_max));
            elseif nr_dimensions==3
                
                joint_density=@(beta1,beta2,beta3) ProbitModelWithResponseError.prior([beta1,beta2,beta3],sigma_p)+...
                    ProbitModelWithResponseError.likelihood(y,X,[beta1,beta2,beta3]);
                
                log_model_evidence=log(integral3(joint_density,beta_min,beta_max,beta_min,beta_max,beta_min,beta_max));
            else
                log_model_evidence=NaN;
            end
            
        end
        
        function [log_model_evidence,posterior_mean,posterior_var,samples]=approximateLogModelEvidence(y,X,sigma_p,epsilon,nr_samples)
            
            prior.log_density=@(betas) ProbitModelWithResponseError.logPrior(betas,sigma_p);
            likelihood.log_density=@(data,betas) ProbitModelWithResponseError.logLikelihood(data,X,betas,epsilon);
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