trial_numbers = unique(trial_index(2:end))
nr_trials = max(trial_numbers)

[h,p,ci,stats]=ttest2(relative_score(PR_type==0 & info_cost==0.01),...
    relative_score(PR_type==1 & info_cost==0.01))

[h,p,ci,stats]=ttest2(relative_score(PR_type==0 & info_cost==1.00),...
    relative_score(PR_type==1 & info_cost==1.00))

[h,p,ci,stats]=ttest2(relative_score(PR_type==0 & info_cost==2.50),...
    relative_score(PR_type==1 & info_cost==2.50))

PR_types = unique(PR_type(2:end));
info_costs = unique(info_cost(2:end));

for ic=1:length(info_costs)
    for pr=1:numel(PR_types)
        for t=1:nr_trials
            condition_met = info_cost==info_costs(ic) & PR_type==PR_types(pr) & trial_index==t;
            avg_rel_score_by_trial(t,pr,ic)=mean(relative_score(condition_met));
            sem_rel_score_by_trial(t,pr,ic)=sem(relative_score(condition_met));
            
            avg_nr_clicks_by_trial(t,pr,ic)=mean(n_click(condition_met));
            sem_nr_clicks_by_trial(t,pr,ic)=sem(n_click(condition_met));
        end
    end
end


for ic=1:length(info_costs)
    fig1=figure(1)
    subplot(1,3,ic)
    errorbar(avg_rel_score_by_trial(:,:,ic),sem_rel_score_by_trial(:,:,ic),'LineWidth',3)
    title(['$',num2str(info_costs(ic)),'/click'],'FontSize',18)
    ylabel('Relative Performance','FontSize',16)
    xlabel('Trial Number','FontSize',16)
    
    fig2=figure(2)
    subplot(1,3,ic)
    errorbar(avg_nr_clicks_by_trial(:,:,ic),sem_nr_clicks_by_trial(:,:,ic),'LineWidth',3)
    title(['$',num2str(info_costs(ic)),'/click'],'FontSize',18)
    ylabel('Avg. Nr. Clicks','FontSize',16)
    xlabel('Trial Number','FontSize',16)
    
end
saveas(fig1,'relativePerformance.fig')
saveas(fig1,'relativePerformance.png')
saveas(fig2,'nrClicks.fig')
saveas(fig2,'nrClicks.png')

