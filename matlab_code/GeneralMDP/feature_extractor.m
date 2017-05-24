function f=feature_extractor(st,a,mdp,selected_features)

if not(exist('selected_features','var'))
    selected_features=1:4;
end

% st_m = reshape(st,2,nr_arms)';
st_m = st;
er = max( st_m(:,1) ./ sum(st_m,2));

if a == mdp.nr_arms+1
    f = [0,0,er,1]';
else
    if ismember(2,selected_features)
        voc1 = VOC1MultiArmBernoulli(st_m(:,1),st_m(:,2),a,mdp.cost);
    else
        voc1=NaN;
    end

    if ismember(1,selected_features)
        vpi = valueOfPerfectInformationMultiArmBernoulli(st_m(:,1),st_m(:,2),a);
    else
        vpi=NaN;
    end
    f = [vpi, voc1, er, 1]';
end

f = f(selected_features,1);

end