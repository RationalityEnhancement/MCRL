classdef BetaPrior
    properties
        alpha %sucesses
        beta  %failures
    end
    methods
        function obj=BetaPrior(alpha0,beta0)
            obj.alpha=alpha0;
            obj.beta=beta0;
        end
        function obj=learn(obj,x)
            successes=sum(x==1);
            n=length(x);
            obj.alpha=obj.alpha+successes;
            obj.beta=obj.beta+n-successes;
        end
        function mode=getMode(obj)
            mode=(obj.alpha-1)/(obj.alpha+obj.beta-2);
        end
        function mean=getMean(obj)
            mean=obj.alpha/(obj.alpha+obj.beta);
        end
        function surprise=getSurprise(obj,x)
            surprise=-(obj.alpha-1)*log(x)-(obj.beta-1)*log(1-x)+betaln(obj.alpha,obj.beta);
        end
        
        function var=getVar(obj)
            var=(obj.alpha*obj.beta)/((obj.alpha+obj.beta)^2*(obj.alpha+obj.beta+1));
        end
        
        function var=getLaplaceVar(obj)
            var=(obj.alpha+obj.beta-2)^2/((obj.alpha-1)*(obj.beta-1));
        end
        
        function density=getDensity(obj,x)
            density=x.^(obj.alpha-1).*(1-x).^(obj.beta-1)./beta(obj.alpha,obj.beta);
        end
    end
end