%%  Which experiment do you want to analyze?
%experiment_name='1A';
experiment_name='1B';

%%
if strcmp(experiment_name,'1A')
%import_data: MCRL/experiments/data/1.0A/trials_matlab.csv
MCRL_path='/Users/Falk/Dropbox/PhD/Metacognitive RL/MCRL/';

max_score=load([MCRL_path,'/experiments/data/stimuli/exp1/optimal',experiment_name,'.csv']);
min_score=load([MCRL_path,'/experiments/data/stimuli/exp1/worst',experiment_name,'.csv']);

eval(['import_data_exp',experiment_name])

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
end
%% Plot fits

if strcmp(experiment_name,'1A')

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
handles=barwitherr([sem_test_block_performance([1,2]); sem_test_block_performance([3,4])],...
           [avg_test_block_performance([1,2]); avg_test_block_performance([3,4])]),
hold on
handles(3)=plot([0.5,2.5],mean(rel_score_pi_star(:,2))*ones(1,2),'LineWidth',3)
set(gca,'XTickLabel',{'constant delays','PR-based delays'},'FontSize',16)
legend(handles,'No message','Message','optimal strategy','Location','best')
ylabel('Relative Performance in Test Block','FontSize',16)
title('Experiment 1B','FontSize',18)
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