function HPD_CI=HPDInterval(posterior,probability_mass)
%This function computes the highest-posterior-density credible interval of
%a parameter whose posterior mass function is posterior.pmf and the 
%corresponding parameter values are posterior.values. This function
%assumes that the posterior is smooth and unimodal.
%probability_mass is the total probability mass of the CI.

[~,max_pos]=max(posterior.pmf);

LB_ind=max_pos; UB_ind=max_pos;

probability_of_CI=sum(posterior.pmf(LB_ind:UB_ind));

while probability_of_CI<probability_mass
    
    if UB_ind==length(posterior.values) && LB_ind>1
        LB_ind=LB_ind-1;
    elseif LB_ind==1 && UB_ind<length(posterior.values)
        UB_ind=UB_ind+1;        
    elseif UB_ind==length(posterior.values) && LB_ind==1
        break;
    elseif (posterior.pmf(LB_ind-1)>posterior.pmf(UB_ind+1))
        LB_ind=LB_ind-1;
    else
        UB_ind=UB_ind+1;
    end
    
    probability_of_CI=sum(posterior.pmf(LB_ind:UB_ind));

end

HPD_CI=[posterior.values(LB_ind),posterior.values(UB_ind)];

end