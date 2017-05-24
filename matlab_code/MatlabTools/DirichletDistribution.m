classdef DirichletDistribution
    properties
        alphas %virtual number of observations for each outcome
        mean %alphas/sum(alphas) 
        scale %sum(alphas)
        nr_values
    end
    methods
        
        %Constructor
        function dirichlet_distribution=DirichletDistribution(alphas)
            dirichlet_distribution.alphas=alphas;
            dirichlet_distribution.nr_values=numel(alphas);
        end
        
        function dirichlet_distribution=update(dirichlet_distribution,events)

            event_counts=hist(events,1:dirichlet_distribution.nr_values);
            dirichlet_distribution.alphas=dirichlet_distribution.alphas+event_counts;
        end
        
        function m=getMean(dirichlet_distribution)
            dirichlet_distribution.mean=dirichlet_distribution.alphas/sum(dirichlet_distribution.alphas);
            m=dirichlet_distribution.mean;
        end
        
        function s=getScale(dirichlet_distribution)
            dirichlet_distribution.scale=sum(dirichlet_distribution.alphas);
            s=dirichlet_distribution.scale;
        end
    end
    
    methods (Static)
        function samples=sample(alphas,nr_samples)
            n = length(alphas);
            r = gamrnd(repmat(alphas,nr_samples,1),1,nr_samples,n);
            samples = r ./ repmat(sum(r,2),1,n);
        end
    end
end
            