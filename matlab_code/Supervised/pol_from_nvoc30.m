cost = 0.001;
Q_hat = zeros(s, nr_actions);
pol_nvoc = zeros(s,1);
nvec = X*w;
for i=1:s
    st = S(i,:);
%     nv = nvoc(33-sum(st),st,cost); %sanity check
    nv = nvec(i);
    Q_hat(i,1) = nv + max(st)/sum(st) ;
    Q_hat(i,2) = max(st)/sum(st);
    
    if nv > 0
        pol_nvoc(i) = 1;
    else 
        pol_nvoc(i) = 2;
    end
end
vhat = max(Q_hat,[],2); %approx v by max of approx q
scatter(vhat,values(1:465,1));
varexp = corr(vhat,values(1:465,1));
title(varexp);