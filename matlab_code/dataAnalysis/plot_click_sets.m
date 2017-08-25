BLUE = [0,50,98]/255;
GOLD = [253,181,21]/255;
SAVE = false;
idx_FB = false(size(data));
idx_noFB = true(size(data));
load trial_properties_high_cost_condition.mat

%% plot common sets of click locations, for each trial

locations = ...
    {[.5,.5],[.7,.5],[.5,.7],[.3,.5],[.5,.3],...
    [.9,.5],[.5,.9],[.1,.5],[.5,.1],...
    [.9,.7],[.9,.3],[.3,.9],[.7,.9],[.1,.7],[.1,.3],[.3,.1],[.7,.1]};
% for f = 1:2
dat = data;
%     if f==1,dat=data(idx_FB);else,dat=data(idx_noFB);end
for i = 1:nr_trials
    clear click_fequencies click_fequencies_FB click_fequencies_noFB click_sets
    click_sets{1} = dat(1).click_locations_before_first_move{i};
    click_fequencies(1) = 1;
    click_fequencies_FB(1) = 0;
    click_fequencies_noFB(1) = 0;
    for j = 1:length(dat)
        cur_sequence = dat(j).click_locations_before_first_move{i};
        % make leaf node pairs on the same branch equivalent
        cur_sequence(cur_sequence==11) = 10;
        cur_sequence(cur_sequence==13) = 12;
        cur_sequence(cur_sequence==15) = 14;
        cur_sequence(cur_sequence==17) = 16;
        cur_sequence = sort(cur_sequence);
        plus1 = true;
        for k = 1:length(click_sets)
            if length(click_sets{k})==length(cur_sequence) && all(click_sets{k}==cur_sequence)
                plus1 = false;
                click_fequencies(k) = click_fequencies(k) + 1;
                if ismember(j,find(idx_FB))
                    click_fequencies_FB(k) = click_fequencies_FB(k) + 1;
                elseif ismember(j,find(idx_noFB))
                    click_fequencies_noFB(k) = click_fequencies_noFB(k) + 1;
                else
                    error('somethin aint right')
                end
                %                     break
            end
        end
        if plus1
            click_sets = vertcat(click_sets,cur_sequence);
            click_fequencies(end+1) = 1;
            click_fequencies_FB(end+1) = 1;
            click_fequencies_noFB(end+1) = 1;
        end
    end
    %         [click_fequencies, ix] = sort(click_fequencies,'descend');
    %         click_fequencies_FB = click_fequencies_FB(ix);
    %         click_fequencies_noFB = click_fequencies_noFB(ix);
    click_fequency_diff = abs(click_fequencies_FB/sum(idx_FB) - click_fequencies_noFB/sum(idx_noFB));
    click_fequency_diffSign = click_fequencies_FB/sum(idx_FB) - click_fequencies_noFB/sum(idx_noFB);
    [click_fequency_diff, ix] = sort(click_fequency_diff,'descend');
    click_fequency_diffSign = click_fequency_diffSign(ix);
    click_fequencies = click_fequency_diff;
    figure('position',[0,0,450,450]);
    for h = 1:4
        subplot(2,2,h); hold on; axis off;
        xlim([-.05 1.05]); ylim([-.05 1.05])
        if click_fequencies(h) < 1/max([sum(idx_FB),sum(idx_noFB)])
            break
        end
        %             title(['frequency: ',sprintf('%0.2f',click_fequencies(h)/nr_subj)]);
        title(['FB to no FB change: ',sprintf('%0.2f',click_fequency_diffSign(h))]);
        skip = false;
        for l = 2:17
            if skip
                skip = false;
                text(locations{l}(1),locations{l}(2),rew,'horizontalalignment','center')
                continue
            end
            rew = trial_properties(i).reward_by_state(l);
            if rew < 0
                rew = ['-$',num2str(abs(rew))];
            else
                rew = ['$',num2str(rew)];
            end
            if ismember(l,click_sets{ix(h)})
                % indicate equivalence of leaf nodes
                if l == 10 && sum(click_sets{ix(h)}==10)==1
                    plot(locations{10}(1),locations{10}(2),'s','MarkerEdgeColor','k','markersize',30,'MarkerFaceColor',mean([GOLD;BLUE]))
                    plot(locations{11}(1),locations{11}(2),'s','MarkerEdgeColor','k','markersize',30,'MarkerFaceColor',mean([GOLD;BLUE]))
                    skip = true;
                elseif l == 12 && sum(click_sets{ix(h)}==12)==1
                    plot(locations{12}(1),locations{12}(2),'s','MarkerEdgeColor','k','markersize',30,'MarkerFaceColor',mean([GOLD;BLUE]))
                    plot(locations{13}(1),locations{13}(2),'s','MarkerEdgeColor','k','markersize',30,'MarkerFaceColor',mean([GOLD;BLUE]))
                    skip = true;
                elseif l == 14 && sum(click_sets{ix(h)}==14)==1
                    plot(locations{14}(1),locations{14}(2),'s','MarkerEdgeColor','k','markersize',30,'MarkerFaceColor',mean([GOLD;BLUE]))
                    plot(locations{15}(1),locations{15}(2),'s','MarkerEdgeColor','k','markersize',30,'MarkerFaceColor',mean([GOLD;BLUE]))
                    skip = true;
                elseif l == 16 && sum(click_sets{ix(h)}==16)==1
                    plot(locations{16}(1),locations{16}(2),'s','MarkerEdgeColor','k','markersize',30,'MarkerFaceColor',mean([GOLD;BLUE]))
                    plot(locations{17}(1),locations{17}(2),'s','MarkerEdgeColor','k','markersize',30,'MarkerFaceColor',mean([GOLD;BLUE]))
                    skip = true;
                else
                    plot(locations{l}(1),locations{l}(2),'s','MarkerEdgeColor','k','markersize',30,'MarkerFaceColor',GOLD)
                end
                text(locations{l}(1),locations{l}(2),rew,'horizontalalignment','center')
            else %~(skip11 && l==11) || ~(skip13 && l==13) || ~(skip15 && l==15) || ~(skip17 && l==17)
%                 if l == 11 && sum(click_sets{ix(h)}==10)==1
%                     plot(locations{10}(1),locations{10}(2),'s','MarkerEdgeColor','k','markersize',30,'MarkerFaceColor',mean([GOLD;BLUE]))
%                     plot(locations{11}(1),locations{11}(2),'s','MarkerEdgeColor','k','markersize',30,'MarkerFaceColor',mean([GOLD;BLUE]))
%                     skip = true;
%                 elseif l == 13 && sum(click_sets{ix(h)}==12)==1
%                     plot(locations{12}(1),locations{12}(2),'s','MarkerEdgeColor','k','markersize',30,'MarkerFaceColor',mean([GOLD;BLUE]))
%                     plot(locations{13}(1),locations{13}(2),'s','MarkerEdgeColor','k','markersize',30,'MarkerFaceColor',mean([GOLD;BLUE]))
%                     skip = true;
%                 elseif l == 15 && sum(click_sets{ix(h)}==14)==1
%                     plot(locations{14}(1),locations{14}(2),'s','MarkerEdgeColor','k','markersize',30,'MarkerFaceColor',mean([GOLD;BLUE]))
%                     plot(locations{15}(1),locations{15}(2),'s','MarkerEdgeColor','k','markersize',30,'MarkerFaceColor',mean([GOLD;BLUE]))
%                     skip = true;
%                 elseif l == 17 && sum(click_sets{ix(h)}==16)==1
%                     plot(locations{16}(1),locations{16}(2),'s','MarkerEdgeColor','k','markersize',30,'MarkerFaceColor',mean([GOLD;BLUE]))
%                     plot(locations{17}(1),locations{17}(2),'s','MarkerEdgeColor','k','markersize',30,'MarkerFaceColor',mean([GOLD;BLUE]))
%                     skip = true;
%                 else
                    plot(locations{l}(1),locations{l}(2),'s','MarkerEdgeColor','k','markersize',30,'MarkerFaceColor',BLUE)
%                 end
                text(locations{l}(1),locations{l}(2),rew,'horizontalalignment','center','color',[.99,.99,.99])
            end
        end
    end
    if SAVE saveas(gcf,[figdir,'/click_sets_trial',num2str(i),'_diff'],'png');end
end

% end