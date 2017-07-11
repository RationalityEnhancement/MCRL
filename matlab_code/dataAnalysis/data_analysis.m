%import_data: MCRL/experiments/data/1.0A/trials_matlab.csv

MCRL_path='/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/';

max_score=load([MCRL_path,'/experiments/data/stimuli/exp1/optimal1A.csv'])
min_score=load([MCRL_path,'/experiments/data/stimuli/exp1/worst1A.csv'])

info_costs=unique(info_cost);

for i=1:numel(score)
    condition_nr=find(info_costs==info_cost(i));
    relative_score(i,1)=(score(i)-min_score(trial_i(i)+1,condition_nr))/...
        (max_score(trial_i(i)+1,condition_nr)-min_score(trial_i(i)+1,condition_nr));
end

trial_numbers = unique(trial_index(2:end))
nr_trials = max(trial_numbers)

[h,p,ci,stats]=ttest2(relative_score(strcmp(PR_type,'featureBased') & info_cost==0.01),...
    relative_score(strcmp(PR_type,'none') & info_cost==0.01))

[h,p,ci,stats]=ttest2(relative_score(strcmp(PR_type,'featureBased') & info_cost==0.01),...
    relative_score(strcmp(PR_type,'objectLevel') & info_cost==0.01))


[h,p,ci,stats]=ttest2(relative_score(strcmp(PR_type,'featureBased') & info_cost==1.00),...
    relative_score(strcmp(PR_type,'none') & info_cost==1.00))

[h,p,ci,stats]=ttest2(relative_score(strcmp(PR_type,'featureBased') & info_cost==1.00),...
    relative_score(strcmp(PR_type,'objectLevel') & info_cost==1.00))


[h,p,ci,stats]=ttest2(relative_score(strcmp(PR_type,'featureBased') & info_cost==2.50),...
    relative_score(strcmp(PR_type,'none') & info_cost==2.50))

[h,p,ci,stats]=ttest2(relative_score(strcmp(PR_type,'featureBased') & info_cost==2.50),...
    relative_score(strcmp(PR_type,'objectLevel') & info_cost==2.50))


[h,p,ci,stats]=ttest2(relative_score(strcmp(PR_type,'featureBased') & info_cost==2.50 & trial_index>3),...
    relative_score(strcmp(PR_type,'none') & info_cost==2.50 & trial_index > 3))

[h,p,ci,stats]=ttest2(relative_score(strcmp(PR_type,'featureBased') & info_cost==2.50 & trial_index<=3),...
    relative_score(strcmp(PR_type,'none') & info_cost==2.50 & trial_index <= 3))

PR_types = unique(PR_type(2:end));
info_costs = unique(info_cost(2:end));

for ic=1:length(info_costs)
    for pr=1:numel(PR_types)
        for t=1:nr_trials
            condition_met = info_cost==info_costs(ic) & strcmp(PR_type,PR_types(pr)) & trial_index==t;
            avg_rel_score_by_trial(t,pr,ic)=mean(relative_score(condition_met));
            sem_rel_score_by_trial(t,pr,ic)=sem(relative_score(condition_met));
            
            avg_nr_clicks_by_trial(t,pr,ic)=mean(n_click(condition_met));
            sem_nr_clicks_by_trial(t,pr,ic)=sem(n_click(condition_met));
        end
    end
end


rel_score_pi_star=csvread('/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/rel_score_pi_star_1A.csv');
optimal_nr_clicks=csvread('/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/nr_observations_pi_star_1A.csv');

FB_types = {'metacognitive FB','no FB','action FB'};
for ic=1:length(info_costs)
    fig1=figure(1)
    subplot(1,3,ic)
    errorbar(avg_rel_score_by_trial(:,:,ic),sem_rel_score_by_trial(:,:,ic),'LineWidth',3), hold on
    plot([1,12],rel_score_pi_star(ic)*[1,1],'.-','LineWidth',3)
    set(gca,'FontSize',20)
    xlim([0.5,13])
    legend(FB_types,'Location','SouthEast')
    title(['$',num2str(info_costs(ic)),'/click'],'FontSize',32)
    ylabel('Relative Performance','FontSize',24)
    xlabel('Trial Number','FontSize',24)
    
    fig2=figure(2)
    subplot(1,3,ic)
    errorbar(avg_nr_clicks_by_trial(:,:,ic),sem_nr_clicks_by_trial(:,:,ic),'LineWidth',3),hold on
    set(gca,'FontSize',20)
    %plot([1,12],optimal_nr_clicks(ic)*[1,1],'.-','LineWidth',3)
    title(['$',num2str(info_costs(ic)),'/click'],'FontSize',32)
    xlim([0.50,13]),ylim([0,15])
    ylabel('Avg. Nr. Clicks','FontSize',24)
    xlabel('Trial Number','FontSize',24)
    %legend('no FB', 'FB','optimal','Location','SouthEast')
    if ic==1
        legend(FB_types,'Location','SouthEast')
    end
    
end

figure(1),tightfig
figure(2),tightfig

saveas(fig1,'relativePerformance.fig')
saveas(fig1,'relativePerformance.png')
saveas(fig2,'nrClicks.fig')
saveas(fig2,'nrClicks.png')

%% Learning curve analysis, meta-level PRs vs. no FB

comparisons={{'featureBased','none'},...
    {'featureBased', 'objectLevel'},...
    {'objectLevel','none'}};

included(:,1) = or(strcmp(PR_type,'none'),strcmp(PR_type,'featureBased'));
included(:,2) = or(strcmp(PR_type,'objectLevel'),strcmp(PR_type,'featureBased'));
included(:,3) = or(strcmp(PR_type,'none'),strcmp(PR_type,'objectLevel'));

labels={'metaFB_vs_noFB','metaFB_vs_actionFB','actionFB_vs_noFB'};

for c=1:numel(comparisons)
    X=[trial_index(info_cost==0.01 & included(:,c)),...
        strcmp(PR_type(info_cost==0.01 & included(:,c)),comparisons{c}{1})];
    y=relative_score(info_cost==0.01 & included(:,c));
    eval(['model_low_cost_',labels{c},' = fitnlm(X,y,''y ~ (1-b1+b2*x2)*sigmoid(b6+(x1-1)*(b3+b4*x2))+b5+b7*x2'',[0.01;0.01;0.25;0.1;0.5;0;0.1])'])
    fit_low(:,1,c)=eval(['model_low_cost_',labels{c},'.predict([(1:12)'',zeros(12,1)])'])
    fit_low(:,2,c)=eval(['model_low_cost_',labels{c},'.predict([(1:12)'',ones(12,1)])'])

    
    X=[trial_index(info_cost==1 & included(:,c)),...
        strcmp(PR_type(info_cost==1 & included(:,c)),comparisons{c}{1})];
    y=relative_score(info_cost==1 & included(:,c));
    %X=[[(1:12)';(1:12)'],[ones(12,1); 2*ones(12,1)]]
    %y=[avg_rel_score_by_trial(:,1,2);avg_rel_score_by_trial(:,2,2)];
    eval(['model_medium_cost_',labels{c},' = fitnlm(X,y,''y ~ (1-b1+b2*x2)*sigmoid(b6+x1*(b3+b4*x2))+b5+b7*x2'',[0.2;1;0.25;0.5;1;0.5;0.1])']);
    eval(['linear_model_medium_cost_',labels{c},' = fitnlm(X,y,''y ~ (b1+b2*x2)*x1'',[1; 1])'])

    fit_medium(:,1,c)=eval(['model_medium_cost_',labels{c},'.predict([(1:12)'',zeros(12,1)])'])
    fit_medium(:,2,c)=eval(['model_medium_cost_',labels{c},'.predict([(1:12)'',ones(12,1)])'])

    
    X=[trial_index(info_cost==2.5 & included(:,c)),...
        strcmp(PR_type(info_cost==2.5 & included(:,c)),comparisons{c}{1})];
    y=relative_score(info_cost==2.5 & included(:,c));
    eval(['model_high_cost_',labels{c},' = fitnlm(X,y,''y ~ (1-b1+b5*x2)*sigmoid(b2+x1*(b3+b4*x2))+b6+b7*x2'',[0.01;0.01;0.25;0.1;0.1; 0.5; 0.1])'])
    BIC_high_cost_nonlinear(c)=eval(['model_high_cost_',labels{c},'.ModelCriterion.BIC'])
    eval(['linear_model_high_cost_',labels{c},'=fitnlm(X,y,''y ~ (b1+b3*x2)*x1+b2+b4*x2'',[0.1;0.5;0.1;0.1])'])
    BIC_high_cost_linear(c)=eval(['linear_model_high_cost_',labels{c},'.ModelCriterion.BIC'])
    fit_high(:,1,c)=eval(['model_high_cost_',labels{c},'.predict([(1:12)'',zeros(12,1)])'])
    fit_high(:,2,c)=eval(['model_high_cost_',labels{c},'.predict([(1:12)'',ones(12,1)])'])
        
end

%% Plot fits
for ic=1:length(info_costs)
    fig1=figure(1)
    subplot(1,3,ic)
    errorbar(avg_rel_score_by_trial(:,:,ic),sem_rel_score_by_trial(:,:,ic),'o','MarkerSize',8), hold on
    %plot([1,12],rel_score_pi_star(ic)*[1,1],'.','LineWidth',3)
    set(gca,'FontSize',20)
    xlim([0.5,13])
    
    if ic==1
        plot(1:12,fit_low(:,2,1),'b-','LineWidth',3)
        plot(1:12,fit_low(:,1,1),'r-','LineWidth',3)
        plot(1:12,fit_low(:,2,3),'-','LineWidth',3,'Color',[1 .5 0])
    elseif ic==2
        plot(1:12,fit_medium(:,2,1),'b-','LineWidth',3)
        plot(1:12,fit_medium(:,1,1),'r-','LineWidth',3)
        plot(1:12,fit_medium(:,2,3),'-','LineWidth',3,'Color',[1 .5 0])        
    elseif ic==3
        plot(1:12,fit_high(:,2,1),'b-','LineWidth',3)
        plot(1:12,fit_high(:,1,1),'r-','LineWidth',3)
        plot(1:12,fit_high(:,2,3),'-','LineWidth',3,'Color',[1 .5 0])        
    end
    
    legend(FB_types,'Location','SouthEast')
    title(['$',num2str(info_costs(ic)),'/click'],'FontSize',32)
    ylabel('Relative Performance','FontSize',24)
    xlabel('Trial Number','FontSize',24)
end

figure(1),tightfig

%% Analyze Data from Experiment 1B
message_types=unique(message);
PR_types=unique(PR_type);

MCRL_path='/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/';

max_score=load([MCRL_path,'/experiments/data/stimuli/exp1/optimal.csv']);
min_score=load([MCRL_path,'/experiments/data/stimuli/exp1/worst.csv']);

info_costs=unique(info_cost);

for i=1:numel(score)
    condition_nr=find(info_costs==info_cost(i));
    relative_score(i,1)=(score(i)-min_score(trial_i(i)+1,condition_nr))/...
        (max_score(trial_i(i)+1,condition_nr)-min_score(trial_i(i)+1,condition_nr));
end

conditions=[1,1; 1 2; 2 1; 2 2];
nr_conditions=size(conditions,1);

nr_trials=12;

PR_labels={'feature-based PR','no PR'};
message_labels={'full message','no message'};
for c=1:nr_conditions

    message_value = message_types(conditions(c,1));
    PR_value = PR_types(conditions(c,2));
    
    condition_names{c}=[PR_labels{conditions(c,2)},', ',...
        message_labels{conditions(c,1)}];
    
    for t=1:nr_trials
        in_condition = strcmp(message,message_value) & strcmp(PR_type,PR_value) ...
            & trial_index ==t;
        avg_rel_score(t,c)=mean(relative_score(in_condition));
        sem_rel_score(t,c)=sem(relative_score(in_condition));
    end
end

line_styles={'-','-','--','--'};
line_colors={[0 1 0],[1 0 0],[0 1 0],[1 0 0]};
figure(1),
for c=1:nr_conditions
    errorbar(avg_rel_score(:,c),sem_rel_score(:,c),'LineWidth',3,'LineStyle',line_styles{c},'Color',line_colors{c}),hold on
    set(gca,'FontSize',18)
    xlabel('Trial Number','FontSize',24)
    ylabel('Relative Performance','FontSize',24)
    legend(condition_names)
end