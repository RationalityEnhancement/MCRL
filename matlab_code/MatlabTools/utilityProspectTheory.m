function u=utilityProspectTheory(o,alpha,beta,gamma)

    if o>0
        u=power(o,alpha);
    else
        u=-gamma*power(-o,beta);
    end

end