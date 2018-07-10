nr_sub = 200;

consistent_with_SAT_TTB = zeros(nr_sub,2,2);
consistent_with_SAT_TTB2 = zeros(nr_sub,2,2);
consistent_with_SAT_TTB3 = zeros(nr_sub,2,2);
consistent_with_TTB = zeros(nr_sub,2,2);

for sub = 1:nr_sub
    %     if data.isFullyRevealed(sub)
    for s = [1,0]
        if s==1
            SAT_threshold = 7;
        else
            SAT_threshold = 7;
        end
        stakes_block = find(data.high_stakes(sub,:)==s);
        for d = [1,0]
            dispersion_block = find(data.high_dispersion(sub,stakes_block,:)==d);
            for i = 1:length(dispersion_block)
                if data.isFullyRevealed(sub)
                    payoffs = squeeze(data.payoff_matrices(sub,stakes_block,dispersion_block(i),:,:));
                    probabilities = squeeze(data.probabilities(sub,stakes_block,dispersion_block(i),:));
                    chosen_gamble = data.chosen_gamble(sub,stakes_block,dispersion_block(i));
                    [~,ix] = sort(probabilities);
                    highest_probability = ix(end);
                    second_highest_probability = ix(end-1);
                    third_highest_probability = ix(end-2);
                    if length(find(probabilities==max(probabilities)))>1
                        disp('equal probabilities')
                    end
                    values = payoffs(highest_probability,:);
                    if length(values)==1 || all(values(end)>values(1:end-1))
                        consistent_with_SAT_TTB(sub,abs(d-2),abs(s-2)) = consistent_with_SAT_TTB(sub,abs(d-2),abs(s-2))+1;
                        values2 = payoffs(second_highest_probability,:);
                        if length(values2)==1 || all(values2(end)>values2(1:end-1))
                            consistent_with_SAT_TTB2(sub,abs(d-2),abs(s-2)) = consistent_with_SAT_TTB2(sub,abs(d-2),abs(s-2))+1;
                            values3 = payoffs(third_highest_probability,:);
                            if length(values3)==1 || all(values3(end)>values3(1:end-1))
                                consistent_with_SAT_TTB3(sub,abs(d-2),abs(s-2)) = consistent_with_SAT_TTB3(sub,abs(d-2),abs(s-2))+1;
                            end
                        end
                    end
                    if values(chosen_gamble)==max(values)
                        consistent_with_TTB(sub,abs(d-2),abs(s-2)) = consistent_with_TTB(sub,abs(d-2),abs(s-2))+1;
                    end
                    % consistent_with_TTB = [];
                    % consistent_with_TTBplus = [];
                    % consistent_with_random = [];
                else
                    consistent_with_SAT_TTB(sub,abs(d-2),abs(s-2)) = data.consistent_with_SAT_TTB(sub,abs(d-2),abs(s-2));
                    consistent_with_SAT_TTB2(sub,abs(d-2),abs(s-2)) = data.consistent_with_SAT_TTB2(sub,abs(d-2),abs(s-2));
                    consistent_with_SAT_TTB3(sub,abs(d-2),abs(s-2)) = data.consistent_with_SAT_TTB3(sub,abs(d-2),abs(s-2));
                    consistent_with_TTB(sub,abs(d-2),abs(s-2)) = data.consistent_with_TTB(sub,abs(d-2),abs(s-2));
                end
            end
        end
    end
end

ix = logical(data.isFullyRevealed);
% ix2 = ~data.isFullyRevealed;
SAT_TTB = cat(3,squeeze(mean(consistent_with_SAT_TTB(~ix,:,:)./5,1)),squeeze(mean(consistent_with_SAT_TTB(ix,:,:)./5,1)));
SAT_TTB2 = cat(3,squeeze(mean(consistent_with_SAT_TTB2(~ix,:,:)./5,1)),squeeze(mean(consistent_with_SAT_TTB2(ix,:,:)./5,1)));
SAT_TTB3 = cat(3,squeeze(mean(consistent_with_SAT_TTB3(~ix,:,:)./5,1)),squeeze(mean(consistent_with_SAT_TTB3(ix,:,:)./5,1)));
TTB = cat(3,squeeze(mean(consistent_with_TTB(~ix,:,:)./5,1)),squeeze(mean(consistent_with_TTB(ix,:,:)./5,1)));

strategies = {'SAT_TTB';'SAT_TTB2';'SAT_TTB3';'TTB'};
% conditions = {'HD-HS';'HD-LS';'LD-HS';'LD-LS';};
HD_HS = [squeeze(SAT_TTB(1,1,:))';squeeze(SAT_TTB2(1,1,:))';squeeze(SAT_TTB3(1,1,:))';squeeze(TTB(1,1,:))'];
HD_LS = [squeeze(SAT_TTB(2,1,:))';squeeze(SAT_TTB2(2,1,:))';squeeze(SAT_TTB3(2,1,:))';squeeze(TTB(2,1,:))'];
LD_HS = [squeeze(SAT_TTB(1,2,:))';squeeze(SAT_TTB2(1,2,:))';squeeze(SAT_TTB3(1,2,:))';squeeze(TTB(1,2,:))'];
LD_LS = [squeeze(SAT_TTB(2,2,:))';squeeze(SAT_TTB2(2,2,:))';squeeze(SAT_TTB3(2,2,:))';squeeze(TTB(2,2,:))'];
tableback = table;
clear('table');
T = table(HD_HS,HD_LS,LD_HS,LD_LS,'RowNames',strategies)
%%

conditions = {'HD-HS';'revealed';'HD-LS';'revealed';'LD-HS';'revealed';'LD-LS';'revealed'};

consistent_with_SAT_TTB = [];
consistent_with_SAT_TTB2 = [];
consistent_with_SAT_TTB3 = [];
consistent_with_TTB = [];
consistent_with_TTBplus = [];
consistent_with_random = [];

consistent_with_SAT = [];
consistent_with_WADD = [];
consistent_with_TTB2plus = [];
consistent_with_FFH = [];

for d = [1,0]
    for s = [1,0]
        for r = [0,1]
            cur_dat = (data.high_dispersion==d).*(repmat(data.high_stakes,1,1,10)==s).*(repmat(data.isFullyRevealed,1,2,10)==r);
            tot = sum(cur_dat(:));
            tmp = data.consistent_with_SAT_TTB.*cur_dat;
            consistent_with_SAT_TTB = [consistent_with_SAT_TTB;sum(tmp(:))/tot];
            tmp = data.consistent_with_SAT_TTB2.*cur_dat;
            consistent_with_SAT_TTB2 = [consistent_with_SAT_TTB2;sum(tmp(:))/tot];
            tmp = data.consistent_with_SAT_TTB3.*cur_dat;
            consistent_with_SAT_TTB3 = [consistent_with_SAT_TTB3;sum(tmp(:))/tot];
            tmp = data.consistent_with_TTB.*cur_dat;
            consistent_with_TTB = [consistent_with_TTB;sum(tmp(:))/tot];
            tmp = data.consistent_with_TTBplus.*cur_dat;
            consistent_with_TTBplus = [consistent_with_TTBplus;sum(tmp(:))/tot];
            tmp = data.consistent_with_random.*cur_dat;
            consistent_with_random = [consistent_with_random;sum(tmp(:))/tot];
            
            tmp = data.consistent_with_SAT.*cur_dat;
            consistent_with_SAT = [consistent_with_SAT;sum(tmp(:))/tot];
            tmp = data.consistent_with_WADD.*cur_dat;
            consistent_with_WADD = [consistent_with_WADD;sum(tmp(:))/tot];
            tmp = data.consistent_with_TTB2plus.*cur_dat;
            consistent_with_TTB2plus = [consistent_with_TTB2plus;sum(tmp(:))/tot];
            tmp = data.consistent_with_FFH.*cur_dat;
            consistent_with_FFH = [consistent_with_FFH;sum(tmp(:))/tot];
        end
    end
end

T = table(consistent_with_SAT_TTB,consistent_with_SAT_TTB2,consistent_with_SAT_TTB3,consistent_with_TTB,...
    consistent_with_TTBplus,consistent_with_random,consistent_with_SAT,consistent_with_WADD,consistent_with_TTB2plus,consistent_with_FFH,...
    'RowNames',conditions)



%%


for i=1:1000
x(i) = -log(rand)./(1/20)+25;
end

%     psumsum = 0;
%     cont = true;
%     while (psumsum != 1 || cont){
%         cont = true;
%         psumsum = 0;
%         psum = 0;
%         for (o=0;o<nr_outcomes;o++){
%             prob = 0;
%             while (prob==0){
%                 prob = Math.round(Math.random()*100)/100;
%             }
%             probabilities[o]=prob;
%             psum+=probabilities[o];
%         }
%         psumsum = 0;
%         for (o=0;o<nr_outcomes;o++){
%             if (probabilities[o]<0.01){
%                 cont = true;
%             }
%         }
%         for (o=0;o<nr_outcomes;o++){   
%             probabilities[o] = Math.round(probabilities[o]/psum*100)/100;
%             psumsum+=probabilities[o];
%         }
%         if (isHighCompensatory[block_nr-1][trial_nr-1]){
%             for (o=0;o<nr_outcomes;o++){
%                 if (probabilities[o]>=0.85){
%                     cont = false
%                 }
%             }
%         }
%         else{
%             cont = false
%             for (o=0;o<nr_outcomes;o++){
%                 if (probabilities[o]>=0.4 || probabilities[o]<=0.1){
%                     cont = true
%                 }
%             }
%         }
%     }