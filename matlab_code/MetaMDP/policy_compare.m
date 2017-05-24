%After get_MDPpseudorewards, meta_semi_gradient_SARSA, value_guess

compare = zeros(465,2);

for i = 1:465
    for j = 1:31
        if sum(S(i,:))-1 == j
            compare(i,1) = pol(i);
            compare(i,2) = policy(i,j);
        end
    end
end

c = compare(:,1) == compare(:,2);
sum(c)/465