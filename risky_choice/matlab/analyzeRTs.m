% data.RTs{subj1-100}{trial1-20}{click0min-28max}

click1 = [];
click2plus = [];
for s = 1:100
    for t = 1:20
        RTs = [data.RTs{s}{t}{:}];
        RTs = RTs(RTs~=-1);
        if ~isempty(RTs)
            click1 = [click1,RTs(1)];
        end
        if length(RTs) > 1
            click2plus = [click2plus,RTs(2:end)];
        end
    end
end
figure,hist(click1,100)
title('RTs before first click','fontsize',20)
saveas(gcf,'~/Desktop/RTs_before_first_click.png')
figure,hist(click2plus,100)
title('RTs after first click','fontsize',20)
saveas(gcf,'~/Desktop/RTs_after_first_click.png')
[p,chi2,df,cohens_w] = chi2test({click1,click2plus})
[h,p,ci,stats] = ttest2(click1,click2plus)

%%

click1_low = [];
click2plus_low = [];
click1_hi = [];
click2plus_hi = [];
subj1_low = [];
subj2plus_low = [];
subj1_hi = [];
subj2plus_hi = [];
for s = 1:100
    if strcmp(data.basic_info{s}.block1_stakes,'low-stakes')
        t1 = 1:10;
        t2 = 11:20;
    else
        t2 = 1:10;
        t1 = 11:20;
    end
    for t = t1
        RTs = [data.RTs{s}{t}{:}];
        RTs = RTs(RTs~=-1);
        if ~isempty(RTs)
            click1_low = [click1_low,RTs(1)];
            subj1_low = [subj1_low,s];
        end
        if length(RTs) > 1
            click2plus_low = [click2plus_low,RTs(2:end)];
            subj2plus_low = [subj2plus_low,s*ones(size(RTs(2:end)))];
        end
    end
    for t = t2
        RTs = [data.RTs{s}{t}{:}];
        RTs = RTs(RTs~=-1);
        if ~isempty(RTs)
            click1_hi = [click1_hi,RTs(1)];
            subj1_hi = [subj1_hi,s];
        end
        if length(RTs) > 1
            click2plus_hi = [click2plus_hi,RTs(2:end)];
            subj2plus_hi = [subj2plus_hi,s*ones(size(RTs(2:end)))];
        end
    end
end
click1_low = click1_low(click1_low<30);
click2plus_low = click2plus_low(click2plus_low<30);
click1_hi = click1_hi(click1_hi<30);
click2plus_hi = click2plus_hi(click2plus_hi<30);
subj1_low = subj1_low(click1_low<30);
subj2plus_low = subj2plus_low(click2plus_low<30);
subj1_hi = subj1_hi(click1_hi<30);
subj2plus_hi = subj2plus_hi(click2plus_hi<30);
figure
subplot(2,2,1),hist(click1_low,100)
title('RTs before first click - low stakes','fontsize',20)
xticks([0:30:30,100:100:1000])
subplot(2,2,2),hist(click2plus_low,100)
title('RTs after first click - low stakes','fontsize',20)
xticks([0:30:30,100:100:1000])
subplot(2,2,3),hist(click1_hi,100)
title('RTs before first click - hi stakes','fontsize',20)
xticks([0:30:30,100:100:1000])
subplot(2,2,4),hist(click2plus_hi,100)
title('RTs after first click - hi stakes','fontsize',20)
xticks([0:30:30,100:100:1000])
saveas(gcf,'~/Desktop/RTs_distributions.png')

%%
figure,hold on
b1 = bar(1,mean(click1_low),'facecolor',[0,0,139/255]);
errorbar(1,mean(click1_low),sem(click1_low),'k')
b2 = bar(2,mean(click2plus_low),'facecolor',[70,130,180]./255);
errorbar(2,mean(click2plus_low),sem(click2plus_low),'k')
b3 = bar(3,mean(click1_hi),'facecolor',[128/255,0,0]);
errorbar(3,mean(click1_hi),sem(click1_low),'k')
b4 = bar(4,mean(click2plus_hi),'facecolor',[205/255,92/255,92/255]);
errorbar(4,mean(click2plus_hi),sem(click2plus_hi),'k')
legend([b1,b2,b3,b4],'click 1 low stakes','click 2+ low stakes','click 1 high stakes','click 2+ high stakes')
xticklabels({})
ylabel('RT','fontsize',20)
saveas(gcf,'~/Desktop/RTs.png')

[h,p,ci,stats] = ttest2([click1_low,click1_hi],[click2plus_low,click2plus_hi])
[h,p,ci,stats] = ttest2([click1_low,click2plus_low],[click1_hi,click2plus_hi])
%%
figure,hold on
b1 = bar(1,median(click1_low),'facecolor',[0,0,139/255]);
errorbar(1,median(click1_low),sem(click1_low),'k')
b2 = bar(2,median(click2plus_low),'facecolor',[70,130,180]./255);
errorbar(2,median(click2plus_low),sem(click2plus_low),'k')
b3 = bar(3,median(click1_hi),'facecolor',[128/255,0,0]);
errorbar(3,median(click1_hi),sem(click1_low),'k')
b4 = bar(4,median(click2plus_hi),'facecolor',[205/255,92/255,92/255]);
errorbar(4,median(click2plus_hi),sem(click2plus_hi),'k')
legend([b1,b2,b3,b4],'click 1 low stakes','click 2+ low stakes','click 1 high stakes','click 2+ high stakes')
xticklabels({})
ylabel('RT','fontsize',20)
saveas(gcf,'~/Desktop/RTs_median.png')

%%
y = [click1_low click2plus_low click1_hi click2plus_hi];
subj = [subj1_low,subj2plus_low,subj1_hi,subj2plus_hi];
click = [ones(size(click1_low)) 2*ones(size(click2plus_low)) ones(size(click1_hi)) 2*ones(size(click2plus_hi))];
stakes = [ones(size(click1_low)) ones(size(click2plus_low)) 2*ones(size(click1_hi)) 2*ones(size(click2plus_hi))];
[p, tt, stats, terms] = anovan(y(:),{subj(:) click(:) stakes(:)}, ...
        'model', 'interaction', ... %try full
        'display', 'on', ...
        'random', [1], ... %random effects for subjects
        'varnames', {'subject' 'click 1 vs 2+' 'stakes low vs high'});
