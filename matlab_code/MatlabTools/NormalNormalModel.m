classdef NormalNormalModel
    properties
    end
    methods (Static)
        function [mu_posterior,tau_posterior]=learn(mu_prior,tau_prior,tau_likelihood,mean_observations,nr_observations)
            tau_posterior=tau_prior+nr_observations.*tau_likelihood;
            mu_posterior=1./tau_posterior.*(tau_prior.*mu_prior+tau_likelihood.*nr_observations.*mean_observations);
        end
        
        function [mu_posterior_diff,tau_posterior_diff]=posteriorBeliefAboutDifference(mu_prior,tau_prior,tau_likelihood,mean1,n1,mean2,n2)
            [mu_posterior1,tau_posterior1]=NormalNormalModel.learn(mu_prior,tau_prior,tau_likelihood,mean1,n1);
            [mu_posterior2,tau_posterior2]=NormalNormalModel.learn(mu_prior,tau_prior,tau_likelihood,mean2,n2);
            
            mu_posterior_diff=mu_posterior1-mu_posterior2;
            tau_posterior_diff=1./(1./tau_posterior1+1./tau_posterior2);
            
        end
    end
end