%import_data: MCRL/experiments/data/0.6/trials_matlab.csv


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


rel_score_pi_star=csvread('/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/rel_score_pi_star.csv');
optimal_nr_clicks=csvread('/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/nr_observations_pi_star.csv');

for ic=1:length(info_costs)
    fig1=figure(1)
    subplot(1,3,ic)
    errorbar(avg_rel_score_by_trial(:,:,ic),sem_rel_score_by_trial(:,:,ic),'LineWidth',3), hold on
    plot([1,12],rel_score_pi_star(ic)*[1,1],'.-','LineWidth',3)
    set(gca,'FontSize',20)
    xlim([0.5,13])
    legend('no FB','FB','optimal','Location','SouthEast')
    title(['$',num2str(info_costs(ic)),'/click'],'FontSize',32)
    ylabel('Relative Performance','FontSize',24)
    xlabel('Trial Number','FontSize',24)
    
    fig2=figure(2)
    subplot(1,3,ic)
    errorbar(avg_nr_clicks_by_trial(:,:,ic),sem_nr_clicks_by_trial(:,:,ic),'LineWidth',3),hold on
    set(gca,'FontSize',20)
    %plot([1,12],optimal_nr_clicks(ic)*[1,1],'.-','LineWidth',3)
    title(['$',num2str(info_costs(ic)),'/click'],'FontSize',32)
    xlim([0.50,13])
    ylabel('Avg. Nr. Clicks','FontSize',24)
    xlabel('Trial Number','FontSize',24)
    %legend('no FB', 'FB','optimal','Location','SouthEast')
    legend('no FB', 'FB','Location','SouthEast')
    
end

figure(1),tightfig
figure(2),tightfig

saveas(fig1,'relativePerformance.fig')
saveas(fig1,'relativePerformance.png')
saveas(fig2,'nrClicks.fig')
saveas(fig2,'nrClicks.png')

X=[trial_index(info_cost==0.01),PR_type(info_cost==0.01)];
y=relative_score(info_cost==0.01);
model_low_cost = fitnlm(X,y,'y ~ (1-b1+b5*x2)*sigmoid(b2+x1*(b3+b4*x2))',[0.01;0.01;0.25;0.2;0.1])

X=[trial_index(info_cost==1),PR_type(info_cost==1)];
y=relative_score(info_cost==1);
model_medium_cost = fitnlm(X,y,'y ~ (1-b1+b5*x2)*sigmoid(b2+x1*(b3+b4*x2))',[0.01;0.01;0.25;0.2;0.1])

X=[trial_index(info_cost==2.5),PR_type(info_cost==2.5)];
y=relative_score(info_cost==2.5);
model_high_cost = fitnlm(X,y,'y ~ (1-b1+b5*x2)*sigmoid(b2+x1*(b3+b4*x2))',[0.01;0.01;0.25;0.2;0.1])
BIC_nonlinear=model_high_cost.ModelCriterion.BIC
linear_model_high_cost=fitnlm(X,y,'y ~ (b1+b3*x2)*x1+b2+b4*x2',[0.1;0.5;0.1;0.1])
BIC_linear=linear_model_high_cost.ModelCriterion.BIC
