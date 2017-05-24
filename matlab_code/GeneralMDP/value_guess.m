mdp = generalMDP(4,1);

for i=1:6436
    for a=1:5
        l = reshape(states(i,:),4,2);
        f = feature_extractor(l,a,mdp);
        val_guess(i,a) = dot(w,f);
    end
end

[m,pol] = max(val_guess',[],1);
rse = 1/8009*sum((m-values(1:6436,1)).^2);
scatter(m,values(1:6436,1))

