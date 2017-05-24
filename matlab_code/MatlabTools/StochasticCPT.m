classdef StochasticCPT < CumulativeProspectTheory
    %Stochastic cumulative prospect theory (SCPT)
    %   as defined in Erev et al. (2010), p. 26      
    
    properties        
        mu=2.15
    end
    
    methods
        function scpt=StochasticCPT()
            scpt.alpha=0.89;
            scpt.beta=0.98;
            scpt.lambda=1.5;
            scpt.mu=2.15;
            scpt.gamma=0.7;
            scpt.delta=0.7;
        end
        
        function p_choice=choice_probability(scpt,outcomes,probabilities)
            
            nr_actions=size(outcomes,2);
            pi_of_p=@(p,exponent) power(p,exponent)./...
                power(power(p,exponent)+power(1-p,exponent),1/exponent);
            
            for i=1:size(probabilities,1)
                for k=1:size(probabilities,2)
                    if outcomes(i,k)>0
                        weights(i,k)=pi_of_p(probabilities(i,k),scpt.gamma);
                    else
                        weights(i,k)=pi_of_p(probabilities(i,k),scpt.delta);
                    end
                end
            end
            
            weighted_values=scpt.weighted_value(outcomes,probabilities);
            
            %compute the absolute distance D between the two value
            %distributions
            delta=bsxfun( @(x,y) abs(x-y), outcomes(:,1),outcomes(:,2)');
            w_delta=bsxfun(@(x,y) x*y, weights(:,1),weights(:,2)');            
            D=dot(delta(:),w_delta(:)/sum(w_delta(:)));
            
            %probability to choose the first gamble
            p_choice(1)=sigmoid((weighted_values(1)-weighted_values(2))/(scpt.mu/D));
            p_choice(2)=1-p_choice(1);
            
        end
    end
    
end

