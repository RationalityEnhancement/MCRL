function benefit_of_planning = benefitOfPlanning(mdps)

for m=1:numel(mdps)
    properties(m)=evaluateMouselabMDP(mdps(m).T,mdps(m).rewards,mdps(m).start_state,...
        mdps(m).horizon,false, mdps(m).horizon);
    
    delta_R(m,:)=properties(m).delta_R;
    benefit(m,:) = delta_R(m,:) - delta_R(m,1);
    
    for n=1:3
        n_step_planning_beneficial(m,n)=benefit(m,n+1)>benefit(m,n);
    end
    
end

benefit_of_planning.avg=mean(benefit)';
benefit_of_planning.std=std(benefit)';
benefit_of_planning.p=mean(n_step_planning_beneficial)';

end