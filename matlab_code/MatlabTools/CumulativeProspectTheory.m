classdef CumulativeProspectTheory
    %Cumulative prospect theory for binary gambles as described in Erev, et
    %al. (2010) on pages 24-26 and
    %https://www.le.ac.uk/ec/research/RePEc/lec/leecon/dp10-10.pdf
          
    properties
        alpha=0.7;
        beta=1;
        lambda=1;
        gamma=0.65;
        delta=0.65;
    end
    
    methods
        
        function cpt=CumulativeProspectTheory()

        end
        
        function cpt=set_parameters(alpha,beta,lambda,gamma,delta)
            cpt.alpha=alpha;
            cpt.beta=beta;
            cpt.lambda=lambda;
            cpt.gamma=gamma;
            cpt.delta=delta;
        end
        
        function [p_choice,choice]=choice_probability(cpt,outcomes,probabilities)
            WVs=cpt.weighted_value(outcomes,probabilities);
            
            %Choose actions with maximum weighted value with equal
            %probability and never choose actions with sub-maximal weighted value.
            max_value=max(WVs);
            choice=find(WVs==max_value);
            nr_best_actions=numel(choice);
            
            nr_actions=size(outcomes,2);
            p_choice=zeros(1,nr_actions);
            p_choice(choice)=1/nr_best_actions;            
        end
        
        function WV=weighted_value(cpt,outcomes,probabilities)
            
            nr_actions=size(outcomes,2);
            for a=1:nr_actions
                WV(a)=dot(cpt.value(outcomes(:,a)),cpt.weighting_function(probabilities(:,a),outcomes(:,a)));
            end
        end
        
        function V=value(cpt,outcomes)
            
            for o=1:numel(outcomes)
                if outcomes(o)>=0
                    V(o)=power(outcomes(o),cpt.alpha);
                else
                    V(o)=-cpt.lambda*power(abs(outcomes(o)),cpt.beta);
                end
            end
        end
        
        function outcomes=inverse_value(cpt,values)
            
            for o=1:numel(values)
                if values(o)>=0
                    outcomes(o)=power(values(o),1/cpt.alpha);
                else
                    outcomes(o)=-1/cpt.lambda*power(abs(values(o)),1/cpt.beta);
                end
            end
        end
        
        function weights=weighting_function(cpt,probabilities,outcomes)
            %probability-weighting function of cumulative prospect theory
            %for two possible outcomes:
            %probabilities: column vector of probabilities
            %outcomes: column vector of the outcomes associated with these
            %probabilities
            probability_weighting_function=@(p,exponent) power(p,exponent)./...
                power(power(p,exponent)+power(1-p,exponent),1/exponent);
                        
            gain_positions=find(outcomes>=0);
            loss_positions=find(outcomes<0);
            
            gains=outcomes(gain_positions);
            losses=outcomes(loss_positions);
            gain_probs=probabilities(gain_positions);
            loss_probs=probabilities(loss_positions);
            
            [gain_values,gain_order]=sort(gains,'descend');
            [loss_values,loss_order]=sort(losses,'ascend');
            
            prob_sorted_gains=gain_probs(gain_order);
            prob_sorted_losses=loss_probs(loss_order);
            
            cdf_gains=cumsum(prob_sorted_gains);
            cdf_losses=cumsum(prob_sorted_losses);
            
            weights_cdf_gains=probability_weighting_function(cdf_gains,cpt.gamma);
            weights_cdf_losses=probability_weighting_function(cdf_losses,cpt.delta);
            
            if not(isempty(gain_positions))
                weights_of_sorted_gains=[weights_cdf_gains(1);diff(weights_cdf_gains)];
                gain_weights(gain_order)=weights_of_sorted_gains;
                weights(gain_positions)=gain_weights;
            end
            
            if not(isempty(loss_positions))
                weights_of_sorted_losses=[weights_cdf_losses(1);diff(weights_cdf_losses)];
                loss_weights(loss_order)=weights_of_sorted_losses;
                weights(loss_positions)=loss_weights;
            end
            
        end
    end 
end