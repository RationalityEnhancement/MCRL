%% plot the locations of clicks -- one plot
addpath ~/Dropbox/MatlabFuncs/
clear num_clicks click_locations

locations = ...
    {[.5,.5],[.7,.5],[.5,.7],[.3,.5],[.5,.3],...
    [.9,.5],[.5,.9],[.1,.5],[.5,.1],...
    [.9,.7],[.9,.3],[.3,.9],[.7,.9],[.1,.7],[.1,.3],[.3,.1],[.7,.1]};
cmap = parula(10000);
for f = 1:2
    if f==1,dat=data(idx_FB);else,dat=data(idx_noFB);end
    for i = 1:nr_trials
        click_locations{i} = [];
        for j = 1:length(dat)
            click_locations{i} = [click_locations{i},dat(j).click_locations_before_first_move{i}];
        end
    end
    for i = 1:nr_trials
        for j = 2:17
            num_clicks(j-1,i,f) = sum(click_locations{i}==j)/length(dat);
        end
    end
end
nr_clicks_cmap = num_clicks-min(num_clicks(:));
nr_clicks_cmap = round(10000*(nr_clicks_cmap)/max(nr_clicks_cmap(:)));

figure('position',[0,0,1500,800]);
ax = easy_gridOfEqualFigures([.04,.04,.04,.04,.04,.04], [.01,.01,.08,.01,.08,.01,.08,.01,.01]);
for f = 1:2
    for i = 2:2:2*nr_trials
        sp = i;
        if f==2, sp = i-1; end
        axes(ax(sp))
        hold on;
        axis off
        for j = 2:17
            if nr_clicks(i/2).isSignificant(j-1)
                plot(locations{j}(1),locations{j}(2),'s','linewidth',2,'MarkerEdgeColor','m','markersize',28,'MarkerFaceColor',cmap(max(1,nr_clicks_cmap(j-1,i/2,f)),:))
            else
                plot(locations{j}(1),locations{j}(2),'s','linewidth',.5,'MarkerEdgeColor','k','markersize',28,'MarkerFaceColor',cmap(max(1,nr_clicks_cmap(j-1,i/2,f)),:))
            end
            rew = trial_properties(i/2).reward_by_state(j);
            if rew < 0
                rew = ['-$',num2str(abs(rew))];
            else
                rew = ['$',num2str(rew)];
            end
            text(locations{j}(1),locations{j}(2),rew,'horizontalalignment','center')
        end
    end
end
if SAVE saveas(gcf,[figdir,'/click_locations_before1stFlight'],'png');end
figure('position',[0,0,20,300])
ax = easy_gridOfEqualFigures([.05,.05], [.01,.9]);
axis off
hcb = colorbar;
set(hcb,'YTick',[0,.5,1],'YTickLabel',[sprintf('%0.2f\n',min(num_clicks(:)),mean(num_clicks(:)),max(num_clicks(:)))])
if SAVE saveas(gcf,[figdir,'/click_locations_before1stFlight_colorbar'],'png');end

%% plot the locations of clicks -- one plot per trial
addpath ~/Dropbox/MatlabFuncs/
clear num_clicks

locations = ...
    {[.5,.5],[.7,.5],[.5,.7],[.3,.5],[.5,.3],...
    [.9,.5],[.5,.9],[.1,.5],[.5,.1],...
    [.9,.7],[.9,.3],[.3,.9],[.7,.9],[.1,.7],[.1,.3],[.3,.1],[.7,.1]};
cmap = parula(10000);
for i = 1:nr_trials
    clear num_clicks
    for f = 1:2
        if f==1,dat=data(idx_FB);else,dat=data(idx_noFB);end
        click_locations{i} = [];
        for j = 1:length(dat)
            click_locations{i} = [click_locations{i},dat(j).click_locations_before_first_move{i}];
        end
        for j = 2:17
            num_clicks(j-1,f) = sum(click_locations{i}==j)/length(dat);
        end
    end
    nr_clicks_cmap = num_clicks-min(num_clicks(:));
    nr_clicks_cmap = round(10000*(nr_clicks_cmap)/max(nr_clicks_cmap(:)));
    
    figure('position',[0,0,600,250]);
    ax = easy_gridOfEqualFigures([.08,.229],[.05,.11,.05]);
    for f = 1:2
        if f == 1
%             xlim([0 1.5]); ylim([0 1])
%             subplot(1,10,5:10)
            axes(ax(2))
            title('feedback','fontsize',16)
        else
%             xlim([0 1.5]); ylim([0 1])
%             subplot(1,10,1:4)
            axes(ax(1))
            title('no feedback','fontsize',16)
        end
        hold on;
        axis off
        for j = 2:17
            if nr_clicks(i).isSignificant(j-1)
                plot(locations{j}(1),locations{j}(2),'s','linewidth',2,'MarkerEdgeColor','m','markersize',41,'MarkerFaceColor',cmap(max(1,nr_clicks_cmap(j-1,f)),:))
            else
                plot(locations{j}(1),locations{j}(2),'s','linewidth',.5,'MarkerEdgeColor','k','markersize',41,'MarkerFaceColor',cmap(max(1,nr_clicks_cmap(j-1,f)),:))
            end
            rew = trial_properties(i).reward_by_state(j);
            if rew < 0
                rew = ['-$',num2str(abs(rew))];
            else
                rew = ['$',num2str(rew)];
            end
            text(locations{j}(1),locations{j}(2),rew,'horizontalalignment','center','fontsize',12)
        end
%     if f==1, 
        hcb = colorbar;
        set(hcb,'YTick',[0,.5,1],'YTickLabel',[sprintf('%0.2f\n',min(num_clicks(:)),mean(num_clicks(:)),max(num_clicks(:)))])
%     end
    end
    if SAVE saveas(gcf,[figdir,'/click_locations_before1stFlight_trialID',num2str(i),''],'png');end
end