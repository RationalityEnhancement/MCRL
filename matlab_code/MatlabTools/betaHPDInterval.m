function HPD_CI=betaHPDInterval(alphas,probability_mass)

dx=0.001;
xs=0:dx:1;
posterior_pdf=betapdf(xs,alphas(1),alphas(2));

[~,max_pos]=max(posterior_pdf);

LB_ind=max_pos; UB_ind=max_pos;
LB=xs(LB_ind); UB=xs(UB_ind);

probability_of_CI=betacdf(UB,alphas(1),alphas(2))-betacdf(LB-dx,alphas(1),alphas(2));

while probability_of_CI<probability_mass
    
    if UB_ind==length(xs) && LB_ind>1
        LB_ind=LB_ind-1;
    elseif LB_ind==1 && UB_ind<length(xs)
        UB_ind=UB_ind+1;        
    elseif UB_ind==length(xs) && LB_ind==1
        break;
    elseif (posterior_pdf(LB_ind-1)>posterior_pdf(UB_ind+1))
        LB_ind=LB_ind-1;
    else
        UB_ind=UB_ind+1;
    end
    
    LB=xs(LB_ind); UB=xs(UB_ind);    
    probability_of_CI=betacdf(UB,alphas(1),alphas(2))-betacdf(LB-dx,alphas(1),alphas(2));

end

HPD_CI=[LB,UB];

end