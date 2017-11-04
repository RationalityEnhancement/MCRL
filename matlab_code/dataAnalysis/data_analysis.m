%%  Which experiment do you want to analyze?
experiment_name='1A';
version = '2';
%experiment_name='1B';
%experiment_name='1C';
%version = '1';

MCRL_path='/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/';
fig_dir=[MCRL_path,'matlab_code/dataAnalysis/figures/']

eval(['import_data_exp',experiment_name,'_v',version])

%%
if strcmp(experiment_name,'1A')
    %import_data: MCRL/experiments/data/1.0A/trials_matlab.csv
    
    nr_training_trials = 10;
    training_trials=1:nr_training_trials;
    test_trials=11:16;
    
    max_score=load([MCRL_path,'/experiments/data/stimuli/exp1/optimal',experiment_name,'.',version,'.csv']);
    min_score=load([MCRL_path,'/experiments/data/stimuli/exp1/worst',experiment_name,'.',version,'.csv']);
    
    rel_score_pi_star=load([MCRL_path,'/experiments/data/stimuli/exp1/rel_score_pi_star',experiment_name,'.',version,'.csv']);
    nr_observations_pi_star=load([MCRL_path,'/experiments/data/stimuli/exp1/nr_observations_pi_star',experiment_name,'.',version,'.csv']);
    
    optimal_nr_clicks=mean(nr_observations_pi_star)';
    optimal_performance=mean(rel_score_pi_star)'-mean(rel_score_pi_star)';
    
    PR_types = unique(PR_type);
    PR_types=PR_types([2,3,1]);
    
    info_costs=unique(info_cost);
    
    for i=1:numel(score)
        condition_nr=find(info_costs==info_cost(i));
        %{
        relative_score(i,1)=(score(i)-min_score(trial_id(i)+1,condition_nr))/...
            (max_score(trial_id(i)+1,condition_nr)-min_score(trial_id(i)+1,condition_nr));
        %}
        relative_score(i,1)=score(i)-max_score(trial_id(i)+1,condition_nr);        
    end
    
    trial_numbers = unique(trial_index(2:end));
    nr_trials = max(trial_numbers);
    
    anovan(relative_score,{PR_type,info_cost},'varnames',{'FB','cost'},'model','interaction')
    
    
    for ic=1:numel(info_costs)
        [h_optimal_vs_none(ic),p_optimal_vs_none(ic),ci_optimal_vs_none(:,ic),stats_optimal_vs_none{ic}]=...
            ttest2(relative_score(strcmp(PR_type,'featureBased') & info_cost==info_costs(ic) & ismember(trial_index,test_trials)),...
            relative_score(strcmp(PR_type,'none') & info_cost==info_costs(ic) & ismember(trial_index,test_trials)))
        
        [h_optimal_vs_action(ic),p_optimal_vs_action(ic),ci_optimal_vs_action(:,ic),stats_optimal_vs_action{ic}]=...
            ttest2(relative_score(strcmp(PR_type,'featureBased') & info_cost==info_costs(ic) & ismember(trial_index,test_trials)),...
            relative_score(strcmp(PR_type,'objectLevel') & info_cost==info_costs(ic) & ismember(trial_index,test_trials)))
        
        [h_action_vs_none(ic),p_action_vs_none(ic),ci_action_vs_none(:,ic),stats_action_vs_none{ic}]=...
            ttest2(relative_score(strcmp(PR_type,'objectLevel') & info_cost==info_costs(ic) & ismember(trial_index,test_trials)),...
            relative_score(strcmp(PR_type,'none') & info_cost==info_costs(ic) & ismember(trial_index,test_trials)))
        
        [h_pi_star_vs_MC_FB(ic),p_pi_star_vs_MC_FB(ic),ci_pi_star_vs_MC_FB,stats_pi_star_vs_MC_FB(ic)]=...
            ttest(relative_score(strcmp(PR_type,'featureBased') & info_cost==info_costs(ic) & ismember(trial_index,test_trials))-optimal_performance(ic))
        
        %analyze the number of clicks
        [h_optimal_vs_none_clicks(ic),p_optimal_vs_none_clicks(ic),ci_optimal_vs_none_clicks(:,ic),stats_optimal_vs_none_clicks{ic}]=...
            ttest2(n_click(strcmp(PR_type,'featureBased') & info_cost==info_costs(ic) & ismember(trial_index,test_trials)),...
            n_click(strcmp(PR_type,'none') & info_cost==info_costs(ic) & ismember(trial_index,test_trials)))
        
        [h_optimal_vs_action_clicks(ic),p_optimal_vs_action_clicks(ic),ci_optimal_vs_action_clicks(:,ic),stats_optimal_vs_action_clicks{ic}]=...
            ttest2(n_click(strcmp(PR_type,'featureBased') & info_cost==info_costs(ic) & ismember(trial_index,test_trials)),...
            n_click(strcmp(PR_type,'objectLevel') & info_cost==info_costs(ic) & ismember(trial_index,test_trials)))
        
        [h_action_vs_none_clicks(ic),p_action_vs_none_clicks(ic),ci_action_vs_none_clicks(:,ic),stats_action_vs_none_clicks{ic}]=...
            ttest2(n_click(strcmp(PR_type,'objectLevel') & info_cost==info_costs(ic) & ismember(trial_index,test_trials)),...
            n_click(strcmp(PR_type,'none') & info_cost==info_costs(ic) & ismember(trial_index,test_trials)))
        
        for pr=1:numel(PR_types)
            avg_rel_score_test_block(ic,pr)=mean(relative_score(...
                strcmp(PR_type,PR_types{pr}) & info_cost==info_costs(ic) & ...
                ismember(trial_index,test_trials)));
            
            median_rel_score_test_block(ic,pr)=median(relative_score(...
                strcmp(PR_type,PR_types{pr}) & info_cost==info_costs(ic) & ...
                ismember(trial_index,test_trials)))
            
            sem_rel_score_test_block(ic,pr)=sem(relative_score(...
                strcmp(PR_type,PR_types{pr}) & info_cost==info_costs(ic) & ...
                ismember(trial_index,test_trials)));
            
            avg_nr_clicks_test_block(ic,pr)=mean(n_click(...
                strcmp(PR_type,PR_types{pr}) & info_cost==info_costs(ic) & ...
                ismember(trial_index,test_trials)));
            
            sem_nr_clicks_test_block(ic,pr)=sem(n_click(...
                strcmp(PR_type,PR_types{pr}) & info_cost==info_costs(ic) & ...
                ismember(trial_index,test_trials)));
            
        end
    end
    
    fig_test=figure()
    barwitherr([sem_rel_score_test_block,zeros(3,1)],...
        [avg_rel_score_test_block,optimal_performance]), hold on
    set(gca,'XTickLabel',{['$',num2str(info_costs(1),2),'/click'],...
        ['$',num2str(info_costs(2),2),'/click'],['$',num2str(info_costs(3),2),'/click']},'FontSize',16),
    xlabel('Time cost','FontSize',18)
    ylabel('Relative performance','FontSize',18)
    %plot([0.5;3.5],repmat(rel_score_pi_star,[2,1]))
    legend('no FB','action FB','metacognitive FB','\pi_{LC}','Location','SouthWest')
    title('Test Block Performance in Exp 1A','FontSize',18)
    saveas(fig_test,[fig_dir,'test_block_performance_Exp1A.png'])
    saveas(fig_test,[fig_dir,'test_block_performance_Exp1A.fig'])
    
    [h,p,ci,stats]=ttest2(relative_score(strcmp(PR_type,'featureBased') & info_cost==info_costs(end) & trial_index>3),...
        relative_score(strcmp(PR_type,'none') & info_cost==info_costs(end) & trial_index > 3))
    
    [h,p,ci,stats]=ttest2(relative_score(strcmp(PR_type,'featureBased') & info_cost==info_costs(end) & trial_index<=3),...
        relative_score(strcmp(PR_type,'none') & info_cost==info_costs(end) & trial_index <= 3))
    
    fig_clicks=figure()
    barwitherr([sem_nr_clicks_test_block,zeros(3,1)],...
        [avg_nr_clicks_test_block,mean(nr_observations_pi_star)'+0.05]), hold on
    %ylim([-1,10])
    set(gca,'XTickLabel',{['$',num2str(info_costs(1),2),'/click'],...
        ['$',num2str(info_costs(2),2),'/click'],['$',num2str(info_costs(3),2),'/click']},'FontSize',16),
    xlabel('Time cost','FontSize',18)
    ylabel('Number Clicks','FontSize',18)
    %plot([0.5;3.5],repmat(rel_score_pi_star,[2,1]))
    legend('no FB','action FB','metacognitive FB','\pi_{LC}')%
    title('Test Block Performance in Exp 1A','FontSize',18)
    saveas(fig_clicks,'figures/test_block_nr_clicks_Exp1A.png')
    
    anovan(n_click,{PR_type,info_cost},'varnames',{'FB','cost'},'model','interaction')
    
    [h,p,ci,stats]=ttest2(n_click(strcmp(PR_type,'featureBased') & info_cost==info_costs(end) & trial_index> nr_training_trials),...
        n_click(strcmp(PR_type,'none') & info_cost==info_costs(end) & trial_index > nr_training_trials))
    
    [h,p,ci,stats]=ttest2(n_click(strcmp(PR_type,'featureBased') & info_cost==info_costs(end) & trial_index> nr_training_trials),...
        n_click(strcmp(PR_type,'objectLevel') & info_cost==info_costs(end) & trial_index > nr_training_trials))
    
    
    PR_types = unique(PR_type(2:end));
    info_costs = unique(info_cost(2:end));
    
    for ic=1:length(info_costs)
        for pr=1:numel(PR_types)
            for t=1:nr_trials
                condition_met = info_cost==info_costs(ic) & strcmp(PR_type,PR_types(pr)) & trial_index==t;
                avg_rel_score_by_trial(t,pr,ic)=mean(relative_score(condition_met));
                median_rel_score_by_trial(t,pr,ic)=median(relative_score(condition_met));
                sem_rel_score_by_trial(t,pr,ic)=sem(relative_score(condition_met));

                avg_score_by_trial(t,pr,ic)=mean(score(condition_met));
                sem_score_by_trial(t,pr,ic)=sem(score(condition_met));                
                
                avg_nr_clicks_by_trial(t,pr,ic)=mean(n_click(condition_met));
                sem_nr_clicks_by_trial(t,pr,ic)=sem(n_click(condition_met));
            end
        end
    end
    
    
    %rel_score_pi_star=csvread(['/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/rel_score_pi_star_1A','.',version,'.csv']);
    %optimal_nr_clicks=csvread(['/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/experiments/data/stimuli/exp1/nr_observations_pi_star_1A','.',version,'.csv']);
    fig_performance=figure()
    fig_nr_clicks=figure()
    FB_types = {'metacognitive FB','no FB','action FB'};
    for ic=1:length(info_costs)
        fig1=figure(fig_performance)
        subplot(1,3,ic)
        errorbar(avg_rel_score_by_trial(:,:,ic),sem_rel_score_by_trial(:,:,ic),'LineWidth',3), hold on
        plot([1,nr_trials],optimal_performance(ic)*[1,1],'.-','LineWidth',3)
        set(gca,'FontSize',20)
        xlim([0.5,nr_trials+1])
        if ic==1
            legend(FB_types,'Location','SouthEast')
        end
        %legend(FB_types,'Location','SouthEast')
        title(['$',num2str(info_costs(ic)),'/click'],'FontSize',32)
        ylabel('Relative Performance','FontSize',24)
        xlabel('Trial Number','FontSize',24)
        
        fig2=figure(fig_nr_clicks)
        subplot(1,3,ic)
        errorbar(avg_nr_clicks_by_trial(:,:,ic),sem_nr_clicks_by_trial(:,:,ic),'LineWidth',3),hold on
        set(gca,'FontSize',20)
        plot([1,nr_trials],optimal_nr_clicks(ic)*[1,1],'.-','LineWidth',3)
        title(['$',num2str(info_costs(ic)),'/click'],'FontSize',32)
        xlim([0.50,nr_trials+1]),ylim([0,12])
        ylabel('Avg. Nr. Clicks','FontSize',24)
        xlabel('Trial Number','FontSize',24)
        %legend('no FB', 'FB','optimal','Location','SouthEast')
        if ic==1
            legend({FB_types{:},'\pi_{LC}'},'Location','SouthEast')
        end
        
    end
    
    figure(fig_performance),tightfig
    figure(fig_nr_clicks),tightfig
    
    saveas(fig_performance,[fig_dir,'relativePerformance.fig'])
    saveas(fig_performance,[fig_dir,'relativePerformance.png'])
    saveas(fig_nr_clicks,[fig_dir,'nrClicks.fig'])
    saveas(fig_nr_clicks,[fig_dir,'nrClicks.png'])
end
%% Learning curve analysis for Experiment 1A: meta-level PRs vs. no FB vs. action FB

if strcmp(experiment_name,'1A')
    
    comparisons={{'featureBased','none'},...
        {'featureBased', 'objectLevel'},...
        {'objectLevel','none'}};
    
    included(:,1) = or(strcmp(PR_type,'none'),strcmp(PR_type,'featureBased'));
    included(:,2) = or(strcmp(PR_type,'objectLevel'),strcmp(PR_type,'featureBased'));
    included(:,3) = or(strcmp(PR_type,'none'),strcmp(PR_type,'objectLevel'));
    
    labels={'metaFB_vs_noFB','metaFB_vs_actionFB','actionFB_vs_noFB'};
    
    comp_idx=[1 2; 1 3; 3 2];
    
    cost_labels={'low','medium','high'};
    
    %for c=1:numel(comparisons)
    c=1;
        opts=statset('nlinfit');
        %opts.Robust='on';
        %opts.RobustWgtFun = 'bisquare';
        
        %X=[trial_index(info_cost==info_costs(1) & included(:,c)),...
        %    strcmp(PR_type(info_cost==info_costs(1) & included(:,c)),comparisons{c}{1})];
        
        for cost_index=1:3
            
            %y=relative_score(info_cost==info_costs(cost_index) & included(:,c));
            
            for t=1:16
                for i=1:3
                    X1_tilde(t,i)=t;
                    X2_tilde(t,i)=i==1; %is meta FB
                    %X3_tilde(t,i)=ismember(i,[1,3]); %is FB
                    X3_tilde(t,i)=i==3;
                end
            end
            X_mean=[X1_tilde(:),X2_tilde(:),X3_tilde(:)];
            %y_temp=avg_rel_score_by_trial(:,[comp_idx(c,1),comp_idx(c,2)],cost_index);
            y_temp=avg_rel_score_by_trial(:,:,cost_index)
            y_mean=y_temp(:);
            X_simple=X_mean(:,1:2);
            X_null=X_mean(:,1);
            
            eval(['complex_model_',cost_labels{cost_index},'_cost',' = fitnlm(X_mean,y_mean,''y_mean ~ (1-b1+b7*x_mean2+b8*x_mean3)*sigmoid(b5+(x_mean1-1)*(b3+b4*x_mean2+b9*x_mean3))+b6+b2*x_mean2+b10*x_mean3'',[0;0;0;0;0;0;0;0;0;0],''Options'',opts)'])
            eval(['model_',cost_labels{cost_index},'_cost',' = fitnlm(X_simple,y_mean,''y_mean ~ (1-b1+b7*x_simple2)*sigmoid(b5+(x_simple1-1)*(b3+b4*x_simple2))+b6+b2*x_simple2'',[0;0;0;0;0;0;0],''Options'',opts)'])
            eval(['null_model_',cost_labels{cost_index},'_cost',' = fitnlm(X_null,y_mean,''y_mean ~ (1-b1)*sigmoid(b5+(x_null-1)*b3)+b6'',[0;0;0;0],''Options'',opts)'])
            
            eval(['fit_',cost_labels{cost_index},'(:,1,c)=model_',cost_labels{cost_index},'_cost','.predict([(1:10)'',zeros(10,1)])'])
            eval(['fit_',cost_labels{cost_index},'(:,2,c)=model_',cost_labels{cost_index},'_cost','.predict([(1:10)'',ones(10,1)])'])
            eval(['fit_',cost_labels{cost_index},'(:,3,c)=model_',cost_labels{cost_index},'_cost','.predict([(1:10)'',zeros(10,1)])'])
            
            BICs_null_model(cost_index)=eval(['null_model_',cost_labels{cost_index},'_cost.ModelCriterion.BIC']);
            BICs_complex_model(cost_index)=eval(['complex_model_',cost_labels{cost_index},'_cost.ModelCriterion.BIC']);
            BICs_model(cost_index)=eval(['model_',cost_labels{cost_index},'_cost.ModelCriterion.BIC']);
        end

        BICs=[BICs_model;BICs_null_model;BICs_complex_model];
        %{
        X=[trial_index(info_cost==1 & included(:,c)),...
            strcmp(PR_type(info_cost==info_costs(2) & included(:,c)),comparisons{c}{1})];
        y=relative_score(info_cost==info_costs(2) & included(:,c));
        %X=[[(1:12)';(1:12)'],[ones(12,1); 2*ones(12,1)]]
        %y=[avg_rel_score_by_trial(:,1,2);avg_rel_score_by_trial(:,2,2)];
        eval(['model_medium_cost_',labels{c},' = fitnlm(X,y,''y ~ (1-b1+b2*x2)*sigmoid(b5+(x1-1)*(b3+b4*x2)+b6*x2)'',[0.01;0.01;0.1;0.1;0;0],''Options'',opts)']);
        eval(['linear_model_medium_cost_',labels{c},' = fitnlm(X,y,''y ~ (b1+b2*x2)*x1'',[1; 1])'])
        
        fit_medium(:,1,c)=eval(['model_medium_cost_',labels{c},'.predict([(1:16)'',zeros(16,1)])'])
        fit_medium(:,2,c)=eval(['model_medium_cost_',labels{c},'.predict([(1:16)'',ones(16,1)])'])
        
        
        X=[trial_index(info_cost==info_costs(3) & included(:,c)),...
            strcmp(PR_type(info_cost==info_costs(3) & included(:,c)),comparisons{c}{1})];
        y=relative_score(info_cost==info_costs(3) & included(:,c));
        eval(['model_high_cost_',labels{c},' = fitnlm(X,y,''y ~ (1-b1+b2*x2)*sigmoid(b5+(x1-1)*(b3+b4*x2)+b6*x2)'',[0.01;0.01;0.1;0.1;0;0],''Options'',opts)'])
        BIC_high_cost_nonlinear(c)=eval(['model_high_cost_',labels{c},'.ModelCriterion.BIC'])
        eval(['linear_model_high_cost_',labels{c},'=fitnlm(X,y,''y ~ (b1+b3*x2)*x1+b2+b4*x2'',[0.1;0.5;0.1;0.1])'])
        BIC_high_cost_linear(c)=eval(['linear_model_high_cost_',labels{c},'.ModelCriterion.BIC'])
        fit_high(:,1,c)=eval(['model_high_cost_',labels{c},'.predict([(1:16)'',zeros(16,1)])'])
        fit_high(:,2,c)=eval(['model_high_cost_',labels{c},'.predict([(1:16)'',ones(16,1)])'])
        %}
    %end
end
%% Plot fits

nr_training_trials = numel(training_trials)

fig_LC_performance=figure()

if strcmp(experiment_name,'1A')
    
    for ic=1:length(info_costs)
        figure(fig_LC_performance)
        subplot(1,3,ic)
        errorbar(avg_rel_score_by_trial(1:nr_training_trials,:,ic),sem_rel_score_by_trial(1:nr_training_trials,:,ic),'o','MarkerSize',8), hold on
        plot([1,nr_training_trials],optimal_performance(ic).*[1,1],'.-','LineWidth',2),hold on
        set(gca,'FontSize',20)
        xlim([0.5,nr_training_trials+0.5])        
        
        if ic==1
            plot(1:nr_training_trials,fit_low(:,2,1),'b-','LineWidth',3),hold on
            plot(1:nr_training_trials,fit_low(:,1,1),'r-','LineWidth',3),hold on
            plot(1:nr_training_trials,fit_low(:,3,1),'-','LineWidth',3,'Color',[1 .5 0]),hold on
            legend({FB_types{:},'\pi_{LC}'},'Location','SouthEast')
        elseif ic==2
            plot(1:nr_training_trials,fit_medium(:,2,1),'b-','LineWidth',3),hold on
            plot(1:nr_training_trials,fit_medium(:,1,1),'r-','LineWidth',3),hold on
            plot(1:nr_training_trials,fit_medium(:,3,1),'-','LineWidth',3,'Color',[1 .5 0]),hold on
            %ylim([0,1.2])
        elseif ic==3
            plot(1:nr_training_trials,fit_high(:,2,1),'b-','LineWidth',3),hold on
            plot(1:nr_training_trials,fit_high(:,1,1),'r-','LineWidth',3),hold on
            plot(1:nr_training_trials,fit_high(:,3,2),'-','LineWidth',3,'Color',[1 .5 0]),hold on
            %ylim([-1.55,1])
        end

        title(['$',num2str(info_costs(ic)),'/click'],'FontSize',32)
        ylabel('Relative Performance','FontSize',24)
        xlabel('Trial Number','FontSize',24)
    end
    
    figure(fig_LC_performance),tightfig
end

saveas(fig_LC_performance,[fig_dir,'LearningCurves1A.fig'])
saveas(fig_LC_performance,[fig_dir,'LearningCurves1A.png'])

%% analyze number of clicks on the first trial before vs. after the first message
for a=1:numel(action_times)
    eval(['move_time(a,:)=',action_times{a},';'])
    eval(['click_time{a}=',strrep(click_times{a},'None','NaN'),';'])
    
end

first_trials = find(trial_index == 1);

for i=1:numel(first_trials)
    idx = first_trials(i);
    nr_clicks_trial1(i)=numel(click_time{idx});
    nr_clicks_before_first_msg(i) = sum(click_time{idx}<move_time(idx,1));
    nr_clicks_after_first_msg(i) = sum(click_time{idx}>move_time(idx,1));
    FB_condition(i) = find(strcmp(PR_type{idx},PR_types))
    cost_condition(i) = find(info_cost(idx)==info_costs)
end

anovan(nr_clicks_trial1,{FB_condition,cost_condition},'model','full','varnames',{'FB','cost'})
anovan(nr_clicks_before_first_msg,{FB_condition,cost_condition},'model','full','varnames',{'FB','cost'})
anovan(nr_clicks_after_first_msg,{FB_condition,cost_condition},'model','full','varnames',{'FB','cost'})

for c=1:3
    avg_nr_clicks_before_msg(c)=mean(nr_clicks_before_first_msg(cost_condition==c))
    avg_nr_clicks_after_msg(c)=mean(nr_clicks_after_first_msg(cost_condition==c))
end
%% Analyze Data from Experiment 1B

if strcmp(experiment_name,'1B')
    
    MCRL_path='/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/';
    
    experiment_name='1B';
    
    max_score=load([MCRL_path,'/experiments/data/stimuli/exp1/optimal',experiment_name,'.csv']);
    min_score=load([MCRL_path,'/experiments/data/stimuli/exp1/worst',experiment_name,'.csv']);
    
    score_pi_star=csvread([MCRL_path,'experiments/data/stimuli/exp1/score_pi_star.csv']);
    rel_score_pi_star=(score_pi_star-min_score)./(max_score-min_score);
    
    nr_observations_pi_star=csvread([MCRL_path,'experiments/data/stimuli/exp1/nr_observations_pi_star.csv']);
    
    eval(['import_data_exp',experiment_name])
    
    message_types={'simple','full'};%unique(message); message_types=message_types(2:-1:1);
    PR_types=unique(PR_type); PR_types=PR_types(2:-1:1);
    
    info_costs=unique(info_cost);
    
    for i=1:numel(score)
        condition_nr=find(info_costs==info_cost(i));
        relative_score(i,1)=(score(i)-min_score(trial_i(i)+1,condition_nr))/...
            (max_score(trial_i(i)+1,condition_nr)-min_score(trial_i(i)+1,condition_nr));
    end
    
    conditions=[1,1; 1 2; 2 1; 2 2];
    nr_conditions=size(conditions,1);
    
    nr_trials=16;
    
    PR_labels={'no PR','feature-based PR'};
    message_labels={'no message','full message'};
    test_trials=11:16;
    for c=1:nr_conditions
        
        message_value = message_types(conditions(c,2));
        PR_value = PR_types(conditions(c,1));
        
        condition_names{c}=[PR_labels{conditions(c,1)},', ',...
            message_labels{conditions(c,2)}];
        
        for t=1:nr_trials
            in_condition = strcmp(message,message_value) & strcmp(PR_type,PR_value) ...
                & trial_index ==t;
            avg_rel_score(t,c)=mean(relative_score(in_condition));
            sem_rel_score(t,c)=sem(relative_score(in_condition));
            
            avg_nr_clicks(t,c)=mean(n_click(in_condition));
            sem_nr_clicks(t,c)=sem(n_click(in_condition));
        end
        
        avg_test_block_performance(c)=mean(avg_rel_score(test_trials,c));
        sem_test_block_performance(c)=sqrt(sum(sem_rel_score(test_trials,c).^2)/numel(test_trials)^2);
        
        avg_nr_clicks_test_block(c)=mean(avg_nr_clicks(test_trials,c));
        sem_nr_clicks_test_block(c)=sem(avg_nr_clicks(test_trials,c));
    end
    
    optimal_nr_clicks_by_trial_type=nr_observations_pi_star(:,2);
    optimal_nr_clicks=optimal_nr_clicks_by_trial_type(trial_i+1);
    
    [h,p,ci,stats]=ttest(n_click(trial_index==16 & strcmp(PR_type,'featureBased') & strcmp(message,'full'))-n_click(trial_index==10 & strcmp(PR_type,'featureBased') & strcmp(message,'full')))
    
    [h,p,ci,stats]=ttest(n_click(trial_index==10 & strcmp(PR_type,'featureBased') & strcmp(message,'full'))-optimal_nr_clicks(trial_index==10 & strcmp(PR_type,'featureBased') & strcmp(message,'full')))
    [h,p,ci,stats]=ttest(n_click(trial_index==16 & strcmp(PR_type,'featureBased') & strcmp(message,'full'))-optimal_nr_clicks(trial_index==10 & strcmp(PR_type,'featureBased') & strcmp(message,'full')))
    
    
    training_trials=1:10;
    
    line_styles={'--','-','--','-'};
    marker_colors={[1 1 1],[1 0 0],[1 1 1],[0 1 0]};
    line_colors={[1 0 0],[1 0 0],[0 1 0],[0 1 0]};
    
    for c=1:nr_conditions
        figure(1),
        errorbar(avg_rel_score(training_trials,c),sem_rel_score(training_trials,c),...
            'LineWidth',3,'LineStyle',line_styles{c},'Color',line_colors{c}),hold on
        set(gca,'FontSize',18)
        xlabel('Trial Number','FontSize',24)
        ylabel('Relative Performance','FontSize',24)
        
        
        figure(2)
        errorbar(avg_nr_clicks(training_trials,c),sem_nr_clicks(training_trials,c),...
            'LineWidth',3,'LineStyle',line_styles{c},'Color',line_colors{c}),hold on
        set(gca,'FontSize',18)
        xlabel('Trial Number','FontSize',24)
        ylabel('Number Clicks','FontSize',24)
        
    end
    fig1=figure(1),
    plot(training_trials,mean(rel_score_pi_star(:,2))*ones(size(training_trials)),'LineWidth',3)
    legend([condition_names(1,:)';'Optimal Strategy'],'Location','SouthEast')
    saveas(fig1,[MCRL_path,'matlab_code/dataAnalysis/figures/AvgRelScore',experiment_name,'.png'])
    
    fig2=figure(2),
    plot(training_trials,mean(nr_observations_pi_star(:,2))*ones(size(training_trials)),'LineWidth',3)
    legend([condition_names(1,:)';'Optimal Strategy'],'Location','SouthEast')
    saveas(fig2,[MCRL_path,'matlab_code/dataAnalysis/figures/AvgNrClicks',experiment_name,'.png'])
    
    
    fig_all_scores=figure(),  fig_all_clicks=figure()
    for c=1:nr_conditions
        figure(fig_all_scores),
        errorbar(avg_rel_score(:,c),sem_rel_score(:,c),...
            'LineWidth',3,'LineStyle',line_styles{c},'Color',line_colors{c}),hold on
        set(gca,'FontSize',18)
        xlabel('Trial Number','FontSize',24)
        ylabel('Relative Performance','FontSize',24)
        
        
        figure(fig_all_clicks)
        errorbar(avg_nr_clicks(:,c),sem_nr_clicks(:,c),...
            'LineWidth',3,'LineStyle',line_styles{c},'Color',line_colors{c}),hold on
        set(gca,'FontSize',18)
        xlabel('Trial Number','FontSize',24)
        ylabel('Number Clicks','FontSize',24)
        
    end
    figure(fig_all_scores),
    plot(1:nr_trials,mean(rel_score_pi_star(:,2))*ones(1,nr_trials),'LineWidth',3)
    xlim([0.5,nr_trials+0.5])
    legend([condition_names(1,:)';'Optimal Strategy'],'Location','SouthEast')
    saveas(fig_all_scores,[MCRL_path,'matlab_code/dataAnalysis/figures/AvgRelScoreAllTrials',experiment_name,'.png'])
    
    figure(fig_all_clicks),
    plot(1:nr_trials,mean(nr_observations_pi_star(:,2))*ones(1,nr_trials),'LineWidth',3)
    xlim([0.5,nr_trials+0.5])
    legend([condition_names(1,:)';'Optimal Strategy'],'Location','SouthEast')
    saveas(fig_all_clicks,[MCRL_path,'matlab_code/dataAnalysis/figures/AvgNrClicksAllTrials',experiment_name,'.png'])
    
    
    
    in_test_block=trial_index>10;
    
    
    message_nr = 0* strcmp(message,message_types{1}) + 1* strcmp(message,message_types{2});
    FB_nr = 0* strcmp(PR_type,PR_types{1}) + 1* strcmp(PR_type,PR_types{2});
    p_FB_test=anovan(relative_score(in_test_block),{FB_nr(in_test_block),...
        message_nr(in_test_block)},'varnames',{'Delay','Message'},'model','full');
    
    
    p_clicks_test=anovan(n_click(in_test_block),{FB_nr(in_test_block),...
        message_nr(in_test_block)},'varnames',{'Delay','Message'},'model','full');
    
    [h,p,ci,stats]=ttest(n_click(in_test_block & strcmp(PR_type,'featureBased'))-optimal_nr_clicks(in_test_block & strcmp(PR_type,'featureBased')))
    
    [h,p,ci,stats]=ttest(n_click(in_test_block & ~strcmp(PR_type,'featureBased'))-optimal_nr_clicks(in_test_block & ~strcmp(PR_type,'featureBased')))
    
    
    fig=figure()
    %xlim([0.6,2]),ylim([0,0.9])
    handles(1,1)=barwitherr([0,sem_test_block_performance(1)],[0,0.875],...
        [0,avg_test_block_performance(1)],'BarWidth',0.2),hold on
    handles(2,1)=barwitherr([0,sem_test_block_performance(3)],[0,1.125],...
        [0,avg_test_block_performance([3])],'BarWidth',0.2),hold on
    handles(1,2)=barwitherr([0,sem_test_block_performance([1])],[0,1.575],...
        [0,avg_test_block_performance([2])],'BarWidth',0.2),hold on
    handles(2,2)=barwitherr([0,sem_test_block_performance([3])],[0,1.825],...
        [0,avg_test_block_performance([4])],'BarWidth',0.2),hold on
    set(gca,'FontSize',16,'XTickLabel',{})
    set(handles(1,1),'FaceColor',[1 1 1],'BarWidth',0.2), set(handles(1,2),'FaceColor',[1 1 1],'BarWidth',0.2)
    set(handles(2,1),'FaceColor',[1 1 1],'BarWidth',0.2) ,set(handles(2,2),'FaceColor',[1 1 1],'BarWidth',0.2)
    set(handles(1,1),'LineStyle','--','LineWidth',3,'EdgeColor',[1 0 0],'BarWidth',0.2),
    set(handles(1,2),'LineStyle','-','LineWidth',3,'EdgeColor',[1 0 0],'BarWidth',0.2),
    set(handles(2,1),'LineStyle','--','LineWidth',3,'EdgeColor',[0 1 0],'BarWidth',0.2)
    set(handles(2,2),'LineStyle','-','LineWidth',3,'EdgeColor',[0 1 0],'BarWidth',0.2)
    set(handles(1,1),'BarWidth',0.1)
    set(handles(1,2),'BarWidth',0.05)
    set(handles(2,1),'BarWidth',0.1)
    set(handles(2,2),'BarWidth',0.05)
    xlim([0.75,2])
    ylim([0.3,0.9])
    handles(3,1)=plot([0.5,2.5],mean(rel_score_pi_star(:,2))*ones(1,2),'LineWidth',3)
    %handles(3)=plot([0.5,2.5],mean(rel_score_pi_star(:,2))*ones(1,2),'LineWidth',3)
    %set(gca,'XTickLabel',{'constant delays','PR-based delays'},'FontSize',16)
    %legend([handles(1,:),handles(2,:)],...
    %    'no PR, no message','no PR, message','PR, no message','PR, message','Location','best')
    ylabel('Relative Performance in Test Block','FontSize',18)
    %title('Experiment 1B','FontSize',18)
    saveas(fig,'figures/TestPerformance1B.png')
    
    
    fig=figure()
    handles=barwitherr([sem_nr_clicks_test_block([1,2]); sem_nr_clicks_test_block([3,4])],...
        [avg_nr_clicks_test_block([1,2]); avg_nr_clicks_test_block([3,4])]),
    hold on
    handles(3)=plot([0.5,2.5],mean(nr_observations_pi_star(:,2))*ones(1,2),'LineWidth',3)
    set(gca,'XTickLabel',{'constant delays','PR-based delays'},'FontSize',16)
    legend(handles,'No message','Message','optimal strategy','Location','North')
    ylabel('Number Clicks in Test Block','FontSize',16)
    title('Experiment 1B','FontSize',18)
    saveas(fig,'figures/NrClicksTestBlock1B.png')
    
    
    % Learning Curves
    comparisons={PR_types,message_types};
    DVs=[strcmp(PR_type,'featureBased'),strcmp(message,'full')];
    
    labels={'PRs','messages','all_vs_nothing'};
    
    is_training_trial=trial_index<=max(training_trials);
    
    with_PR=[false,false,true,true]';
    with_message=[false,true,false,true]';
    DVs=[with_PR,with_message,with_PR & with_message];
    
    %{
for c=1:numel(comparisons)
    
    X=[repmat(training_trials(:),[4,1]),[repmat(0,[2*numel(training_trials),1]);repmat(1,[2*numel(training_trials),1])]];
    avg_scores=[avg_rel_score(training_trials,DVs(:,c)==0),avg_rel_score(training_trials,DVs(:,c)==1)];
    y=avg_scores(:);
    %X=[[(1:12)';(1:12)'],[ones(12,1); 2*ones(12,1)]]
    %y=[avg_rel_score_by_trial(:,1,2);avg_rel_score_by_trial(:,2,2)];
    eval(['model_',labels{c},' = fitnlm(X,y,''y ~ (1-b1)*sigmoid(x1*(b2+b3*x2))'',[0.2;0.25;0.5])']);
    eval(['linear_model_',labels{c},' = fitnlm(X,y,''y ~ (b1+b2*x2)*x1'',[1; 1])'])

    fit(:,1,c)=eval(['model_',labels{c},'.predict([training_trials'',zeros(numel(training_trials),1)])'])
    fit(:,2,c)=eval(['model_',labels{c},'.predict([training_trials'',ones(numel(training_trials),1)])'])

    
    avg_clicks=[avg_nr_clicks(training_trials,DVs(:,c)==0),avg_nr_clicks(training_trials,DVs(:,c)==1)];
    y=avg_clicks(:);
    %X=[[(1:12)';(1:12)'],[ones(12,1); 2*ones(12,1)]]
    %y=[avg_rel_score_by_trial(:,1,2);avg_rel_score_by_trial(:,2,2)];
    eval(['click_model_',labels{c},' = fitnlm(X,y,''y ~ (1-b1)*sigmoid(x1*(b2+b3*x2))'',[0.2;0.25;0.5])']);
    eval(['linear_click_model_',labels{c},' = fitnlm(X,y,''y ~ (b1+b2*x2)*x1'',[1; 1])'])

    fit_nr_clicks(:,1,c)=eval(['click_model_',labels{c},'.predict([training_trials'',zeros(numel(training_trials),1)])'])
    fit_nr_clicks(:,2,c)=eval(['click_model_',labels{c},'.predict([training_trials'',ones(numel(training_trials),1)])'])
    
end
    %}
    
    nr_training_trials=numel(training_trials);
    PR_factor=repmat(with_PR',[nr_training_trials,1]);
    message_factor=repmat(with_message',[nr_training_trials,1]);
    
    X=[repmat(training_trials(:),[4,1]), PR_factor(:), message_factor(:)];
    training_scores=avg_rel_score(training_trials,:);
    y=training_scores(:);
    model=fitnlm(X,y,'y ~ (1-b1+b2*x2+b3*x3)*sigmoid((b4+b5*x2+b6*x3)*(x1-1))+b7',[0.1,0.1,0.1,0.1,0.1,0.1,0.1])
    
    model_noPR_noMessage = fitnlm(X(:,1),y,'y ~ (1-b1)*sigmoid(b2*(x1-1))+b3',[0.1,0.1,0.1])
    model_PR_noMessage = fitnlm(X(:,[1,2]),y,'y ~ (1-b1+b2*x2)*sigmoid((b3+b4*x2)*(x1-1))+b5',[0.1,0.1,0.1,0.1,0.1])
    model_noPR_Message = fitnlm(X(:,[1,3]),y,'y ~ (1-b1+b2*x2)*sigmoid((b3+b4*x2)*(x1-1))+b5',[0.1,0.1,0.1,0.1,0.1])
    
    model_with_interaction=fitnlm(X,y,'y ~ (1-b1+b2*x2+b3*x3+b9*x2*x3)*sigmoid(b4+(b5+b6*x2+b7*x3+b10*x2*x3)*x1)+b8',[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
    
    
    %lower BIC is better
    BIC_model=model.ModelCriterion.BIC
    BIC_noPR_Message=model_noPR_Message.ModelCriterion.BIC
    BIC_PR_noMessage=model_PR_noMessage.ModelCriterion.BIC
    BIC_noPR_noMessage=model_noPR_noMessage.ModelCriterion.BIC
    BIC_complex=model_with_interaction.ModelCriterion.BIC
    
    z=(model.Coefficients.Estimate(2)-model.Coefficients.Estimate(3))/sqrt(model.Coefficients.SE(2)^2+model.Coefficients.SE(3)^2)
    p=1-normcdf(z)
    
    figure()
    subplot(2,1,1)
    plot(mean(avg_rel_score(training_trials,with_PR),2)),hold on
    plot(mean(avg_rel_score(training_trials,~with_PR),2)),hold on
    legend('with PR','without PR')
    subplot(2,1,2)
    plot(mean(avg_rel_score(:,with_message),2)),hold on
    plot(mean(avg_rel_score(:,~with_message),2)),hold on
    
    figure()
    plot(avg_rel_score(:,with_PR & with_message)),hold on
    plot(avg_rel_score(:,~with_PR & ~with_message))
    
    
    X=[repmat(training_trials(:),[4,1]), PR_factor(:), message_factor(:)];
    number_clicks=avg_nr_clicks(training_trials,:);
    y=number_clicks(:);
    click_model=fitnlm(X,y,'y ~ (1-b1+b2*x2+b3*x3)*sigmoid((b4+b5*x2+b6*x3)*(x1-1))+b7',[0.1,0.1,0.1,0.1,0.1,0.1,0.1])
    
    
    nr_training_trials=numel(training_trials(:));
    c=0;
    for pr=1:2
        for m=1:2
            c=c+1;
            condition(c).with_PR=pr==2;
            condition(c).with_message=m==2;
            
            model_fit_by_condition(:,c)=model.predict([training_trials(:),...
                condition(c).with_PR*ones(nr_training_trials,1),...
                condition(c).with_message*ones(nr_training_trials,1)])
            
            click_model_fit_by_condition(:,c)=click_model.predict([training_trials(:),...
                condition(c).with_PR*ones(nr_training_trials,1),...
                condition(c).with_message*ones(nr_training_trials,1)])
            
        end
    end
    
    line_styles={'--','-','--','-'};
    marker_colors={[1 1 1],[1 0 0],[1 1 1],[0 1 0]};
    line_colors={[1 0 0],[1 0 0],[0 1 0],[0 1 0]};
    
    fig_score=figure(), fig_clicks=figure()
    for c=1:numel(condition)
        figure(fig_score)
        plot(model_fit_by_condition(:,c),'LineStyle',line_styles{c},...
            'LineWidth',3,'Color',line_colors{c}),hold on
        set(gca,'FontSize',16)
        
        figure(fig_clicks)
        plot(click_model_fit_by_condition(:,c),'LineStyle',line_styles{c},...
            'LineWidth',3,'Color',line_colors{c}),hold on
        set(gca,'FontSize',16)
        
    end
    figure(fig_score)
    plot(training_trials,mean(rel_score_pi_star(:,2))*ones(size(training_trials)),'LineWidth',3)
    
    figure(fig_clicks)
    plot(training_trials,mean(nr_observations_pi_star(:,2))*ones(size(training_trials)),'LineWidth',3)
    
    for c=1:numel(condition)
        figure(fig_score)
        plot(avg_rel_score(training_trials,c),'o','Color',line_colors{c},'MarkerFaceColor',marker_colors{c})
        set(gca,'FontSize',16)
        
        figure(fig_clicks)
        plot(avg_nr_clicks(training_trials,c),'o','Color',line_colors{c},'MarkerFaceColor',marker_colors{c})
        set(gca,'FontSize',16)
    end
    figure(fig_score)
    xlim([1,10])
    xlabel('Trial Number','FontSize',18)
    ylabel('Relative Score','FontSize',17)
    legend('no PR, no message','no PR, message','PR, no message',...
        'PR, message','optimal strategy','Location','South')
    saveas(fig_score,'figures/LearningCurvesExp1B.png')
    
    figure(fig_clicks)
    xlim([1,10])
    xlabel('Trial Number','FontSize',18)
    ylabel('Number Clicks','FontSize',17)
    legend('no PR, no message','no PR, message','PR, no message',...
        'optimal strategy','PR, message','Location','South')
    saveas(fig_clicks,'figures/LearningCurvesClicksExp1B.png')
    
end

%% Analyze data from Experiment 1C
if strcmp(experiment_name,'1C')
    nr_trials=16;
    
    training_trials = 1:10;
    test_trials=11:16;
    
    conditions = unique(PR_type);
    
    max_score=load([MCRL_path,'/experiments/data/stimuli/exp1/optimal',experiment_name,'.',version,'.csv']);
    min_score=load([MCRL_path,'/experiments/data/stimuli/exp1/worst',experiment_name,'.',version,'.csv']);
    
    rel_score_pi_star=load([MCRL_path,'/experiments/data/stimuli/exp1/rel_score_pi_star_',experiment_name,'.',version,'.csv'])
    nr_observations_pi_star=load([MCRL_path,'/experiments/data/stimuli/exp1/nr_observations_pi_star_',experiment_name,'.',version,'.csv'])
    
    info_costs=unique(info_cost);
    
    condition_nr=2;
    for i=1:numel(score)
        relative_score(i,1)=(score(i)-min_score(trial_i(i)+1,condition_nr))/...
            (max_score(trial_i(i)+1,condition_nr)-min_score(trial_i(i)+1,condition_nr));
    end
    
    
    for c=1:numel(conditions)
        avg_rel_score_test_block(c)=mean(relative_score(...
            strcmp(PR_type,conditions{c}) & ismember(trial_index,test_trials)));
        
        sem_rel_score_test_block(c)=sem(relative_score(...
            strcmp(PR_type,conditions{c}) & ismember(trial_index,test_trials)));
        
        avg_nr_clicks_test_block(c)=mean(n_click(...
            strcmp(PR_type,conditions{c}) & ismember(trial_index,test_trials)));
        
        sem_nr_clicks_test_block(c)=sem(n_click(...
            strcmp(PR_type,conditions{c}) & ismember(trial_index,test_trials)));
        
    end
    
    
    condition_labels = {'Demonstration','Metacognitive FB','Practice without FB'};
    
    fig_1C=figure()
    barwitherr(sem_rel_score_test_block,avg_rel_score_test_block,'FaceColor',[0.75,0.75,0.75])
    set(gca,'XTickLabel',condition_labels,'FontSize',18)
    ylabel('Relative Score','FontSize',18)
    hold on
    plot([0.5,3.5],repmat(rel_score_pi_star(2),[1,2]),'LineWidth',3)
    title('Test Block Performance in Exp 1C','FontSize',18)
    
    saveas(fig_1C,'figures/PerformanceExp1C.png')
    
    [h,p,ci,stats]=ttest2(relative_score(strcmp(PR_type,'demonstration') & ismember(trial_index,test_trials)),...
        relative_score(strcmp(PR_type,'featureBased') & ismember(trial_index,test_trials)))
    
    [h,p,ci,stats]=ttest2(relative_score(strcmp(PR_type,'featureBased') & ismember(trial_index,test_trials)),...
        relative_score(strcmp(PR_type,'none') & ismember(trial_index,test_trials)))
    
    [h,p,ci,stats]=ttest2(relative_score(strcmp(PR_type,'demonstration') & ismember(trial_index,test_trials)),...
        relative_score(strcmp(PR_type,'none') & ismember(trial_index,test_trials)))
    
    test_scores = relative_score(ismember(trial_index,test_trials));
    test_clicks = n_click(ismember(trial_index,test_trials));
    training_method = PR_type(ismember(trial_index,test_trials))
    anovan(test_scores, {training_method})
    anovan(test_clicks, {training_method})
    
    fig_1C_clicks=figure()
    barwitherr(sem_nr_clicks_test_block,avg_nr_clicks_test_block,'FaceColor',[0.75,0.75,0.75])
    set(gca,'XTickLabel',condition_labels,'FontSize',18)
    ylabel('Number Clicks','FontSize',18)
    hold on
    plot([0.5,3.5],repmat(nr_observations_pi_star(2),[1,2]),'LineWidth',3)
    title('Test Block of Exp 1C','FontSize',18)
    
    saveas(fig_1C_clicks,'figures/ClicksExp1C.png')
    
    [h,p,ci,stats]=ttest2(n_click(strcmp(PR_type,'demonstration') & ismember(trial_index,test_trials)),...
        n_click(strcmp(PR_type,'featureBased') & ismember(trial_index,test_trials)))
    
    [h,p,ci,stats]=ttest2(n_click(strcmp(PR_type,'featureBased') & ismember(trial_index,test_trials)),...
        n_click(strcmp(PR_type,'none') & ismember(trial_index,test_trials)))
    
    [h,p,ci,stats]=ttest2(n_click(strcmp(PR_type,'demonstration') & ismember(trial_index,test_trials)),...
        n_click(strcmp(PR_type,'none') & ismember(trial_index,test_trials)))
    
end

%% Analyze Experiment 1A.2
%1. import data manually
import_data_exp1A2

experiment_name = '1A';
version = '2';
MCRL_path='/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/';

nr_trials = 16;
nr_training_trials = 10;

training_trials = 1:nr_training_trials;
test_trials = (nr_training_trials+1):nr_trials;

max_score=load([MCRL_path,'/experiments/data/stimuli/exp1/optimal',experiment_name,'.',version,'.csv']);
min_score=load([MCRL_path,'/experiments/data/stimuli/exp1/worst',experiment_name,'.',version,'.csv']);

%rel_score_pi_star=load([MCRL_path,'/experiments/data/stimuli/exp1/rel_score_pi_star_',experiment_name,'.',version,'.csv'])
%nr_observations_pi_star=load([MCRL_path,'/experiments/data/stimuli/exp1/nr_observations_pi_star_',experiment_name,'.',version,'.csv'])

info_costs=unique(info_cost);

condition_nr=3;
for i=1:numel(score)
    relative_score(i,1)=(score(i)-min_score(trial_i(i)+1,condition_nr))/...
        (max_score(trial_i(i)+1,condition_nr)-min_score(trial_i(i)+1,condition_nr));
end


for t=1:nr_trials
    avg_nr_clicks(t)=mean(n_click(trial_index==t));
    sem_nr_clicks(t)=sem(n_click(trial_index==t));
    
    avg_rel_score(t)=mean(relative_score(trial_index==t))
    sem_rel_score(t)=sem(relative_score(trial_index==t))
end

figure()
errorbar(avg_nr_clicks,sem_nr_clicks)
ylabel('Number Clicks','FontSize',18)
xlabel('Trial Number','FontSize',18)

figure()
errorbar(avg_rel_score,sem_rel_score)
ylabel('Relative Score','FontSize',18)
ylabel('Trial Number','FontSize',18)

without_FB.avg_rel_score = avg_rel_score;
without_FB.sem_rel_score = sem_rel_score;
without_FB.avg_nr_clicks = avg_nr_clicks;
without_FB.sem_nr_clicks = sem_nr_clicks;
without_FB.trial_index = trial_index;
without_FB.rel_score = relative_score;
without_FB.nr_clicks = n_click;

keep without_FB
%% Analyze Experiment 1A.3
%1. import data manually
import_data_exp1A3

experiment_name = '1A';
version = '3';
MCRL_path='/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/';

nr_trials = 16;
nr_training_trials = 10;

training_trials = 1:nr_training_trials;
test_trials = (nr_training_trials+1):nr_trials;

max_score=load([MCRL_path,'/experiments/data/stimuli/exp1/optimal',experiment_name,'.',version,'.csv']);
min_score=load([MCRL_path,'/experiments/data/stimuli/exp1/worst',experiment_name,'.',version,'.csv']);

%rel_score_pi_star=load([MCRL_path,'/experiments/data/stimuli/exp1/rel_score_pi_star_',experiment_name,'.',version,'.csv'])
%nr_observations_pi_star=load([MCRL_path,'/experiments/data/stimuli/exp1/nr_observations_pi_star_',experiment_name,'.',version,'.csv'])

info_costs=unique(info_cost);

condition_nr=3;
for i=1:numel(score)
    relative_score(i,1)=(score(i)-min_score(trial_i(i)+1,condition_nr))/...
        (max_score(trial_i(i)+1,condition_nr)-min_score(trial_i(i)+1,condition_nr));
end


for t=1:nr_trials
    avg_nr_clicks(t)=mean(n_click(trial_index==t));
    sem_nr_clicks(t)=sem(n_click(trial_index==t));
    
    avg_rel_score(t)=mean(relative_score(trial_index==t))
    sem_rel_score(t)=sem(relative_score(trial_index==t))
end

figure()
errorbar(avg_nr_clicks,sem_nr_clicks)
ylabel('Number Clicks','FontSize',18)
xlabel('Trial Number','FontSize',18)

figure()
errorbar(avg_rel_score,sem_rel_score)
ylabel('Relative Score','FontSize',18)
ylabel('Trial Number','FontSize',18)

with_FB.avg_rel_score = avg_rel_score;
with_FB.sem_rel_score = sem_rel_score;
with_FB.avg_nr_clicks = avg_nr_clicks;
with_FB.sem_nr_clicks = sem_nr_clicks;
with_FB.trial_index = trial_index;
with_FB.rel_score = relative_score;
with_FB.nr_clicks = n_click;


fig_rel_score=figure()
errorbar(without_FB.avg_rel_score,without_FB.sem_rel_score),hold on
errorbar(with_FB.avg_rel_score,with_FB.sem_rel_score),hold on
ylabel('Relative Score','FontSize',18)
xlabel('Trial Number','FontSize',18)
legend('without FB','with FB')
saveas(fig_rel_score,'figures/relScoreExp1A3.png')

fig_nr_clicks=figure()
errorbar(without_FB.avg_nr_clicks,without_FB.sem_nr_clicks),hold on
errorbar(with_FB.avg_nr_clicks,with_FB.sem_nr_clicks),hold on
ylabel('Number Clicks','FontSize',18)
xlabel('Trial Number','FontSize',18)
legend('without FB','with FB')
saveas(fig_nr_clicks,'figures/nrClicksExp1A3.png')

nr_training_trials = 10;
[h_score,p_score,ci_score,stats_score]=ttest2(with_FB.rel_score(...
    with_FB.trial_index>nr_training_trials),...
    without_FB.rel_score(without_FB.trial_index>nr_training_trials))

[h_clicks,p_clicks,ci_clicks,stats_clicks]=ttest2(...
    with_FB.nr_clicks(with_FB.trial_index>nr_training_trials),...
    without_FB.nr_clicks(test_trials))

%% Compute the delays participants would have experienced in the pilot experiment 0.991
load_PRs_pilot_exp_0d991

conditions = unique(info_cost1);

load PR_thresholds

click_thresholds = [thresholds.low_cost(:),thresholds.med_cost(:),thresholds.high_cost(:)];
nr_locations=16;
%threshold click PRs
for i=1:numel(click_pr)
    
    condition(i)=find(info_cost1(i)==conditions);
    clicks_remaining=nr_locations-(click_num(i)-1);
    
    click_threshold=click_thresholds(clicks_remaining,condition(i));
    
    if click_pr(i)<-click_threshold
        thresholded_click_pr(i)=click_pr(i);
    else
        thresholded_click_pr(i)=0;
    end
end

delay_threshold = 2;
delay_per_point = - 1.5;
delays = delay_per_point * thresholded_click_pr;

%compute the vicarious delay of each trial
participants=unique(pid1);
trial_ids = unique(trial_index1);
for p=1:numel(participants)
    for t=1:numel(trial_ids)
        indices = find(pid1==participants(p) & trial_index1==trial_ids(t));
        
        total_delay(p,t)=sum(delays(indices));
        
        if total_delay(p,t)>delay_threshold
            thresholded_total_delay(p,t)=total_delay(p,t);
        else
            thresholded_total_delay(p,t)=0;
        end
        
        condition_by_trial(p,t)=unique(info_cost1(indices));
        
    end
end
%compute the average vicarious delay by condition
avg_delay_condition=NaN(numel(conditions),1);
for c=1:numel(conditions)
    
    avg_delay_by_condition(c) = mean(thresholded_total_delay(condition_by_trial(:)== conditions(c)));
    std_delay_by_condition(c) = std(thresholded_total_delay(condition_by_trial(:)== conditions(c)))
end

%% Determine 95-percentiles of repsonse time distributions by condition
clear
import_trial_data_exp0d991

for t=1:numel(action_times)
    completion_time(t)=max(eval(action_times{t}))/1000;
end

anova1(log(completion_time),info_cost)

conditions=unique(info_cost);

for c=1:numel(conditions)
    completion_time_percentiles(c,:)=percentile(completion_time(info_cost==conditions(c)),[90,95])
end

percentile(completion_time(:),[90,95])

%%
clear
import_trial_data_exp0d992
import_participant_data_exp0d992

participants=unique(pid1);
finished=strcmp(completed,'True');
finishers=participants(finished);
for p=1:numel(finishers)
    avg_nr_clicks_by_participant(p)=mean(n_click1(pid1==finishers(p)))
end

mean(avg_nr_clicks_by_participant(condition(finished)==0))
sem(avg_nr_clicks_by_participant(condition(finished)==0))

mean(avg_nr_clicks_by_participant(condition(finished)==1))
sem(avg_nr_clicks_by_participant(condition(finished)==1))

figure()
barwitherr([sem(avg_nr_clicks_by_participant(condition(finished)==0)),...
    sem(avg_nr_clicks_by_participant(condition(finished)==1))],...
    [mean(avg_nr_clicks_by_participant(condition(finished)==0)),...
    mean(avg_nr_clicks_by_participant(condition(finished)==1))])
ylabel('Average Nr. Clicks','FontSize',18)
set(gca,'XTickLabel',{'Time Limit','No Time Limit'},'FontSize',18)

[h,p,ci,stats]=ttest2(avg_nr_clicks_by_participant(condition(finished)==0),...
    avg_nr_clicks_by_participant(condition(finished)==1))

[p,stats]=chi2test({avg_nr_clicks_by_participant(condition(finished)==0)<=0,...
    avg_nr_clicks_by_participant(condition(finished)==1)<=0})


figure()
barwitherr([sem(score1(condition(finished)==0)),...
    sem(score1(condition(finished)==1))],...
    [nanmean(score1(condition(finished)==0)),...
    nanmean(score1(condition(finished)==1))])
ylabel('Average Score','FontSize',18)
set(gca,'XTickLabel',{'Time Limit','No Time Limit'},'FontSize',18)


[h,p,ci,stats]=ttest2(score1(condition(finished)==0),...
    score1(condition(finished)==1))

%% analyze data from pilot experiment 
import_data_pilot0d993

is_training=trial_index1<=10;
is_test=trial_index1>10;
is_beginning=trial_index1<=3;

info_costs = unique(info_cost1);

for ic=1:numel(info_costs)
    avg_nr_clicks_training(ic)=mean(n_click1(is_training & info_cost1==info_costs(ic)))
    sem_nr_clicks_training(ic)=sem(n_click1(is_training & info_cost1==info_costs(ic)))

    avg_nr_clicks_start(ic)=mean(n_click1(is_beginning & info_cost1==info_costs(ic)))
    sem_nr_clicks_start(ic)=sem(n_click1(is_beginning & info_cost1==info_costs(ic)))
    
    
    avg_nr_clicks_test(ic)=mean(n_click1(is_test & info_cost1==info_costs(ic)))
    sem_nr_clicks_test(ic)=sem(n_click1(is_test & info_cost1==info_costs(ic)))
    
    avg_nr_clicks(ic)=mean(n_click1(info_cost1==info_costs(ic)))
    sem_nr_clicks(ic)=sem(n_click1(info_cost1==info_costs(ic)))
end

%% 
import_data_pilot0d995
is_training=trial_index<=10;
is_test=trial_index>10;
is_beginning=trial_index<=3;

info_costs = unique(info_cost);

for ic=1:numel(info_costs)
    avg_nr_clicks_training(ic)=mean(n_click(is_training & info_cost==info_costs(ic)))
    sem_nr_clicks_training(ic)=sem(n_click(is_training & info_cost==info_costs(ic)))

    avg_nr_clicks_start(ic)=mean(n_click(is_beginning & info_cost==info_costs(ic)))
    sem_nr_clicks_start(ic)=sem(n_click(is_beginning & info_cost==info_costs(ic)))
    
    
    avg_nr_clicks_test(ic)=mean(n_click(is_test & info_cost==info_costs(ic)))
    sem_nr_clicks_test(ic)=sem(n_click(is_test & info_cost==info_costs(ic)))
    
    avg_nr_clicks(ic)=mean(n_click(info_cost==info_costs(ic)))
    sem_nr_clicks(ic)=sem(n_click(info_cost==info_costs(ic)))
end

%% compute the delays participants would have experienced in Pilot 0.995
load_PRs_pilot_exp_0d995

conditions = unique(info_cost1);

load PR_thresholds

click_thresholds = [thresholds.low_cost(:),thresholds.med_cost(:),thresholds.high_cost(:)];
nr_locations=16;
%threshold click PRs
for i=1:numel(click_pr)
    
    condition(i)=find(info_cost1(i)==conditions);
    clicks_remaining=nr_locations-(click_num(i)-1);
    
    click_threshold=click_thresholds(clicks_remaining,condition(i));
    
    if click_pr(i)<-click_threshold
        thresholded_click_pr(i)=click_pr(i);
    else
        thresholded_click_pr(i)=0;
    end
end

delay_threshold = 2;
delay_per_point = - 1.5;
delays = delay_per_point * thresholded_click_pr;

%compute the vicarious delay of each trial
participants=unique(pid1);
trial_ids = unique(trial_index1);
for p=1:numel(participants)
    for t=1:numel(trial_ids)
        indices = find(pid1==participants(p) & trial_index1==trial_ids(t));
        
        total_delay(p,t)=sum(delays(indices));
        
        if total_delay(p,t)>delay_threshold
            thresholded_total_delay(p,t)=total_delay(p,t);
        else
            thresholded_total_delay(p,t)=0;
        end
        
        condition_by_trial(p,t)=unique(info_cost1(indices));
        
    end
end
%compute the average vicarious delay by condition
avg_delay_condition=NaN(numel(conditions),1);
for c=1:numel(conditions)
    
    avg_delay_by_condition(c) = mean(thresholded_total_delay(condition_by_trial(:)== conditions(c)));
    std_delay_by_condition(c) = std(thresholded_total_delay(condition_by_trial(:)== conditions(c)))
end
