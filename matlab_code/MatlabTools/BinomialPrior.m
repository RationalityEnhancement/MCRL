classdef BinomialPrior
    properties
        p=0.5
        n=1
    end
    
    methods
        function obj=BinomialPrior(p0)
            obj.p=p0;
        end
        
        function mode=getMode(obj)
            mode=floor((obj.n+1)*obj.p);
        end
        
        function surprise=getSurprise(obj,x)
            if x==0
                surprise=-log(1-obj.p);
            elseif x==1
                surprise=-log(obj.p);
            end
        end
        
    end
end