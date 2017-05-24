ers = zeros(nr_states,1);
for s=1:nr_states
    st = S(s,:);  
    st_m = reshape(st,2,nr_arms)';
    ers(s) = max(st_m(:,1) ./ sum(st_m,2));
end