classdef LogNormalRandomWalk
    properties
    end
    methods (Static)
        function next=step(from,nr_samples)
            m=0;
            EV=from+m;
            Var=0.01^2;
            
            mu=log(EV.^2./sqrt(Var+EV.^2));
            sigma=sqrt(log(1+Var./EV.^2));
            next=lognrnd(mu,sigma,[nr_samples,1]);
        end
        
        function log_density=logDensity(from,to)
            m=0;
            EV=from+m;
            Var=0.01^2;
            
            mu=log(EV.^2./sqrt(Var+EV.^2));
            sigma=sqrt(log(1+Var./EV.^2));
            
            log_density=log(max(eps,lognpdf(to,mu,sigma)));
            
        end
    end
end