classdef GeneralizedProbitModelWithResponseError
    
    properties
        alpha=0.04225;
        beta=0.065;
        sigma_p=10;
        nr_parameters;
        nr;
        epsilon=0.05;
    end
    
    methods
        function m=GeneralizedProbitModelWithResponseError(model_nr,sigma_p,epsilon)
            
            m.nr=model_nr;
            if exist('sigma_p','var')
                m.sigma_p=sigma_p;
            end
            
            if exist('epsilon','var')
                m.epsilon=epsilon;
            end
            
            if m.nr==8
                m.nr_parameters=2;
            elseif m.nr==9
                m.nr_parameters=3;
            end
        end
        
        function p=predict(model,beta,theta)
            X=createRegressors(model,theta,69);
            p=normcdf(X*beta');
        end
        
        function [X,y]=generateData(model,theta,beta,nr_outcomes)
            
            if not(exist('beta','var'))
                beta=model.sampleBeta(0,1);
            end
            
            if not(exist('theta','var'))
                theta=model.sampleTheta(model.alpha,model.beta);
            end
            
            X=model.createRegressors(theta,nr_outcomes);
            p_y=(1-model.epsilon)*normcdf(X*beta')+model.epsilon*0.5*ones(size(X,1),1);
            y=rand(numel(p_y),1)<=p_y;
        end
        
        function beta=sampleBeta(model,mu,sigma)
            beta=mvnrnd(mu,sigma*diag(ones(size(mu))));
        end
        
        function theta=sampleTheta(model,alpha,beta)
            theta=gamrnd(alpha,1/beta);
        end
        
        function X=createRegressors(model,tau_likelihood,nr_outcomes)
            options=csvread('C:\Users\flieder\Dropbox\PhD\Preference Learning\options_Matlab_friendly.csv');
            nr_subjects=nr_outcomes/size(options,1);
            
            mu_prior=3.62; tau_prior=2.61;
            [mu_Delta,tau_Delta]=NormalNormalModel.posteriorBeliefAboutDifference(mu_prior,tau_prior,tau_likelihood,options(:,1),options(:,2),options(:,3),options(:,4));
            
            if model.nr==8
                X=repmat(sqrt(tau_Delta).*mu_Delta,[nr_subjects,1]);
            elseif model.nr==9
                X=repmat([sqrt(tau_Delta).*mu_Delta,options(:,2)-options(:,4)],[nr_subjects,1]);
            end
        end
        
        function likelihood=Likelihood(model,y,betas,theta)
            X=model.createRegressors(theta);
            likelihood=prod(model.epsilon*0.5+(1-model.epsilon)*normcdf(X(y==1,:)*betas'))*prod(model.epsilon*0.5+(1-model.epsilon)*(1-normcdf(X(y==0,:))*betas'));
        end
        
        function log_likelihood=logLikelihood(model,y,betas,theta)
            log_likelihood=0;
            for i=1:length(theta)
                X=model.createRegressors(theta(i),size(y,1));
                log_likelihood(i)=sum(log(max(eps,model.epsilon*0.5+(1-model.epsilon)*normcdf(X(y==1,:)*betas(i,:)'))))+sum(log(1-min(1-eps,model.epsilon*0.5+(1-model.epsilon)*normcdf(X(y==0,:)*betas(i,:)'))));
            end
        end
        
        function prior_density=prior(model,betas,sigma_p,theta)
            prior_density=prod(normpdf(betas'/sigma_p),1).*...
                gampdf(theta',model.alpha,1/model.beta);
        end
        
        function log_p_betas_theta=logPrior(model,betas,theta)
            log_p_betas_theta=sum(log(max(eps,normpdf(betas'/model.sigma_p))),1)+...
                sum(log(max(eps,gampdf(theta,model.alpha,1/model.beta))));
        end
        
        function estimate=estimateParameters(model,y,sigma_p)
            
            if exist('sigma_p','var') %MAP
                neg_log_joint=@(params) -model.logPrior(params(1:end-1),params(end))-...
                    model.logLikelihood(y,params(1:end-1),params(end));
                
                estimate=fmincon(neg_log_joint,[randn(1,model.nr_parameters-1),gamrnd(model.alpha,1/model.beta)],[zeros(1,model.nr_parameters-1),-1],[-eps],[],[],[],[],[],optimset('algorithm','sqp','Display','iter'));
            else %MLE
                neg_log_likelihood=@(betas_theta) -model.logLikelihood(y,betas_theta(1:end-1),betas_theta(end));
                estimate=fmincon(neg_log_likelihood,[randn(1,model.nr_parameters-1),gamrnd(model.alpha,1/model.beta)],[zeros(1,model.nr_parameters-1),-1],[-eps],[],[],[],[],[],optimset('algorithm','sqp','Display','iter'));
            end
        end
        
        function [BIC,BIC_approximation_to_log_model_evidence,fit]=BIC(model,y)
            MLE=model.estimateParameters(y);
            fit=model.logLikelihood(y,MLE(1:end-1),MLE(end));
            nr_datapoints=size(y,1);
            complexity=model.nr_parameters*log(nr_datapoints);
            BIC=-2*fit+complexity;
            BIC_approximation_to_log_model_evidence=fit-complexity/2;
        end
        
        function [log_model_evidence]=logModelEvidenceNumericalInt(model,y)
            
            nr_dimensions=model.nr_parameters;
            
            beta_min=-10*model.sigma_p; beta_max=10*model.sigma_p;
            theta_min=eps; theta_max=Inf;
            
            if nr_dimensions==1
                joint_density=@(beta,theta) model.prior(beta)+...
                    model.likelihood(y,beta);
                
                log_model_evidence=log(integral(joint_density,beta_min,beta_max));
            elseif nr_dimensions==2
                joint_density=@(beta,theta) model.prior(beta,theta)+...
                    model.likelihood(y,beta,theta);
                log_model_evidence=log(integral2(joint_density,beta_min,beta_max,theta_min,theta_max));
            elseif nr_dimensions==3
                
                joint_density=@(beta1,beta2,theta) model.prior([beta1,beta2],theta)+...
                    model.likelihood(y,[beta1,beta2],theta);
                
                log_model_evidence=log(integral3(joint_density,beta_min,beta_max,beta_min,beta_max,theta_min,theta_max));
            else
                log_model_evidence=NaN;
            end
            
        end
        
        function [log_model_evidence,posterior_mean,posterior_var,samples]=approximateLogModelEvidence(model,y,nr_samples)
            
            prior.log_density=@(betas_theta) model.logPrior(betas_theta(:,1:end-1),betas_theta(:,end));
            likelihood.log_density=@(data,betas_theta) model.logLikelihood(data,betas_theta(:,1:end-1),betas_theta(:,end));
            likelihood.nr_parameters=model.nr_parameters;
            
            temp=model.createRegressors(1,size(y,1));
            proposal.initial_value(1,1:model.nr_parameters-1)=6./range(temp);
            proposal.initial_value(1,model.nr_parameters)=1;
            
            proposal.log_density=@(from,to,sigma) sum(log(max(eps,normpdf((repmat(to(:,1:end-1),[max(size(to,1),size(from,1))/size(to,1),1])-repmat(from(:,1:end-1),[max(size(to,1),size(from,1))/size(from,1),1]))/sigma)))',1)+LogNormalRandomWalk.logDensity(from(:,end),to(:,end))';
            proposal.propose=@(from,sigma) [mvnrnd(from(1:end-1),sigma^2*diag(ones(model.nr_parameters-1,1))),LogNormalRandomWalk.step(from(end),1)];
            proposal.sample=@(from,sigma,nr_samples) [mvnrnd(from(1:end-1),sigma^2*diag(ones(model.nr_parameters-1,1)),nr_samples), LogNormalRandomWalk.step(from(end),nr_samples)];
            proposal.params=min(proposal.initial_value/10);
            
            %betas_theta_MAP=model.estimateParameters(y,model.sigma_p);
            
            [log_model_evidence,~,samples]=...
                lmeChibJeliazkov01(y,prior,likelihood,proposal,nr_samples);
            
            posterior_mean=mean(samples); posterior_var=var(samples);
        end
        
        
    end
end