%Kunz, S. (2009). The Bayesian linear model with unknown variance. Technical report, Seminar for Statistics, ETH Zurich, Switzerland.
%Lindley, D. V. and Smith, A. F. M. (1972). Bayes estimates for the linear model. Journal of the Royal Statistical Society. Series B (Methodological), 34(1).
classdef BayesianGLM
    properties
        %parameters of Gaussian prior on beta given the error precision
        mu_0
        Lambda_0 %p(beta|kappa,mu_0,Lambda_0)=N(mu_0,kappa*Lambda_0)
        sigma
        
        %parameters of Gamma Prior on Error Precision kappa
        a_0=1
        b_0=1
        
        %parameters of the posterior on beta
        mu_n
        Lambda_n
        
        %parameters of the posterior on Error Precision kappa
        a_n
        b_n
        
        nr_regressors
    end
    
    methods
        function obj=setPrior(obj,mu,Lambda,a,b)
            obj.mu_0=mu;
            if exist('Lambda','var')
                obj.Lambda_0=Lambda;
            end
            if and(exist('a','var'),exist('b','var'))
                obj.a_0=a;
                obj.b_0=b;
            end
        end
        
        function obj=BayesianGLM(nr_regressors,sigma)
            
            if exist('nr_regressors','var')
                obj.nr_regressors=nr_regressors;
            else
                obj.nr_regressors=1;
            end
            
            if exist('sigma','var')
                obj.sigma=sigma;
            else
                obj.sigma=1;
            end
            
            mu=zeros(obj.nr_regressors,1);
            Lambda=1/obj.sigma^2*eye(obj.nr_regressors);
            
            obj=obj.setPrior(mu,Lambda,obj.a_0,obj.b_0);
            obj.mu_n=obj.mu_0;
            obj.Lambda_n=obj.Lambda_0;
            obj.a_n=obj.a_0;
            obj.b_n=obj.b_0;
        end
        
        function obj=update(obj,X,y)
            
            %Today's prior is yesterday's posterior.
            obj.Lambda_0=obj.Lambda_n;
            obj.mu_0=obj.mu_n;
            
            n=size(y,1);
            obj.Lambda_n=(X'*X+obj.Lambda_0);
            obj.mu_n=obj.Lambda_n\(X'*y+obj.Lambda_0*obj.mu_0);
            obj.a_n=obj.a_0+n/2;
            obj.b_n=obj.b_0+1/2*(y'*y+obj.mu_0'*obj.Lambda_0*obj.mu_0...
                -obj.mu_n'*obj.Lambda_n*obj.mu_n);
            
            if isnan(rcond(obj.Lambda_n))
                throw(MException('BayesianGLM:rcond','Precision Matrix is ill conditioned!'))
            end
        end       
        
        function obj=computePosterior(obj,X,y)
            %computes the posterior distribution without updating the prior
            n=size(y,1);
            obj.Lambda_n=(X'*X+obj.Lambda_0);
            obj.mu_n=obj.Lambda_n\(X'*y+obj.Lambda_0*obj.mu_0);
            obj.a_n=obj.a_0+n/2;
            obj.b_n=obj.b_0+1/2*(y'*y+obj.mu_0'*obj.Lambda_0*obj.mu_0...
                -obj.mu_n'*obj.Lambda_n*obj.mu_n);            
            
            %{
            pi_epsilon=obj.a_n/obj.b_n;
            obj.Lambda_n=(pi_epsilon*X'*X+obj.Lambda_0);
            obj.mu_n=obj.Lambda_n\(X'*y+obj.Lambda_0*obj.mu_0);
            %}
        end
        
        function obj=learn(obj,X,y)
            %Legacy interface to computePosterior.
            obj=obj.computePosterior(X,y);
        end
        
        function obj=learnAndUpdatePrior(obj,X,y)
            obj=obj.computePosterior(X,y);
            obj.mu_0=obj.mu_n;
            obj.Lambda_0=obj.Lambda_n;
        end
        
        function mll=marginalLikelihood(obj,X,y)
            obj=obj.learn(X,y);
            n=size(y,1);
            mll=(2*pi)^(-n/2)*sqrt(det(obj.Lambda_0)/det(obj.Lambda_n))*...
                gamma(obj.a_n)/gamma(obj.a_0)*...
                obj.b_0^obj.a_0/obj.b_n^obj.a_n;
        end
        
        function log_mll=logMarginalLikelihood(obj,X,y)
            
            obj=obj.learn(X,y);
            n=size(y,1);
            
            log_mll=-n/2*log(2*pi)+...
                1/2*(logdet(obj.Lambda_0)-logdet(obj.Lambda_n))+...
                gammaln(obj.a_n)-gammaln(obj.a_0)+...
                obj.a_0*log(obj.b_0)-obj.a_n*log(obj.b_n);
        end
        
        function [selected_features,p_m]=featureSelection(model,X,y)
            %Efficient Bayesian comparison of nested using
            %Savage-Dickey ratios (Penny, & Ridgway, 2013).
            
            model=model.learn(X,y);
            log_mll=model.logMarginalLikelihood(X,y);
            
            nr_regressors=size(X,2);
            nr_submodels=power(2,nr_regressors)-1;
            weight_is_zero=logical(combn([0,1],nr_regressors));
            weight_is_zero=weight_is_zero(1:end-1,:);
            
            log_BF=zeros(nr_submodels,1);
            log_BF(1)=0; %full model
            
            D=diag(ones(nr_regressors,1));
            
            for m=2:nr_submodels
                
                C=D(:,weight_is_zero(m,:));               
                
                w_0=C'*model.mu_0;
                P_0=inv(C'*inv(model.a_0/model.b_0*model.Lambda_0)*C);

                w_n=C'*model.mu_n;
                P_n=inv(C'*inv(model.a_n/model.b_n*model.Lambda_n)*C);
                
                log_BF(m,1)=1/2*w_n'*P_n*w_n+1/2*(logdet(P_0)-logdet(P_n));
            end
            
            log_evidences=log_mll-log_BF;
            p_m=exp(log_evidences-logsumexp(log_evidences));
            
            [~,best_m]=max(p_m);
            selected_features=find(not(weight_is_zero(best_m,:)));
            
        end
        
        function [betas]=sampleCoefficients(obj,nr_samples)
            if not(exist('nr','var'))
                nr_samples=1;
            end
            
            kappas=max(gamrnd(obj.a_n*ones(nr_samples,1),1/obj.b_n*ones(nr_samples,1)),eps);
            betas=nan(obj.nr_regressors,nr_samples);
            for s=1:nr_samples
                betas(:,s)=mvnrnd_from_Pi(obj.mu_n,kappas(s)*obj.Lambda_n,nr_samples)';                
            end

        end
        
        function [ys,betas,sigmas]=samplePrior(obj,X,nr)
            
            nr_regressors=size(X,2);
            obj=BayesianGLM(nr_regressors);
            
            kappas=max(gamrnd(obj.a_0*ones(nr,1),1/obj.b_0*ones(nr,1)),eps);
            sigmas=1./sqrt(kappas);
            
            betas=nan(nr_regressors,nr);
            ys=nan(size(X,1),nr);
            for i=1:nr
                %Sigma_0=inv(obj.Lambda_0)/kappas(i);
                betas(:,i)=mvnrnd_from_Pi(obj.mu_0,kappas(i)*obj.Lambda_0)';
                ys(:,i)=X*betas(:,i)+sigmas(i)*randn(size(ys(:,i)));
            end
        end
        
        function [ys,betas,sigmas]=samplePosterior(obj,X,y,nr)

            if not(exist('nr','var'))
                nr=1;
            end
            
            if and(exist('X','var'),exist('y','var'))
                nr_regressors=size(X,2);
                obj=BayesianGLM(nr_regressors);
                obj=obj.learn(X,y);
            else
                nr_regressors=numel(obj.mu_n);
            end
            
            kappas=max(gamrnd(obj.a_n*ones(nr,1),1/obj.b_n*ones(nr,1)),eps);
            sigmas=1./sqrt(kappas);
            
            betas=nan(nr_regressors,nr);
            ys=nan(nr_regressors,nr);
            for i=1:nr
                %Pi_n=lambda*Pi_0+lambda*X'*X
                %Sigma_n=inv(obj.Lambda_n)/kappas(i);
                betas(:,i)=mvnrnd_from_Pi(obj.mu_n,kappas*obj.Lambda_n,nr)';
                
                if exist('X','var')
                    ys(:,i)=X*betas(:,i)+sigmas(i)*randn(size(ys(:,1)));
                else
                    ys=[];
                end
            end
            
        end
        
        function y_hat=predict(model,X)
            y_hat=X*model.mu_n;
        end
        
        function model=setParameter(model,parameter_name,parameter_value)
            model.(parameter_name)=parameter_value;
        end
    end
    
    methods (Static)
        function sigma=setSigmaPrior(range_of_y,effective_nr_regressors,E_X2)
            if not(exist('E_X2','var'))
                E_X2=1;
            end
            sigma=range_of_y/(1.96*sqrt(effective_nr_regressors*E_X2));
        end
    end
end