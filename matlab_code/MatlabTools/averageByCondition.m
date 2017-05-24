function [averages,sems]=averageByCondition(values,condition_names)

values=values(:);
conditions=unique(condition_names);

for c=1:numel(conditions)
    condition=conditions(c);
    averages(c)=nanmean(values(condition_names==condition));
    sems(c)=sem(values(condition_names(:)==condition(:)));
end

end