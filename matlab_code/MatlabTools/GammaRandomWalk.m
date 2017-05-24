classdef GammaRandomWalk
    properties
    end
    methods (Static)
        function next=step(from,nr_samples)
            mu=0; sigma=sqrt(0.1);
            EV=from+mu;
            Var=sigma^2;
            
            alpha=EV^2/Var;
            beta=EV/Var;
            next=gamrnd(alpha,1/beta,[nr_samples,1]);
        end
        
        function log_density=logDensity(from,to)
            mu=0; sigma=sqrt(0.1);
            EV=from+mu;
            Var=sigma.^2;
            
            alpha=EV.^2./Var;
            beta=EV./Var;
            log_density=log(max(eps,gampdf(to,alpha,1./beta)));
            
        end
    end
end