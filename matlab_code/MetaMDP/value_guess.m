%After get_MDPpseudorewards, meta_semi_gradient_SARSA
mdp = metaMDP(2,1);

val_guess = zeros(465,2);
for i=1:465
    for a=1:2
        f = feature_extractor(S(i,:)',a,mdp);
        val_guess(i,a) = dot(w,f); % can replace w with w1 or w2
    end
end
[m,pol] = max(val_guess',[],1);
mse = 1/465*sum((m'-values(1:465,1)).^2)
scatter(m,values(1:465,1))