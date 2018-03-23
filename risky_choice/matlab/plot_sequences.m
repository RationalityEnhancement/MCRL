% clear
% load ~/Dropbox/mouselab_cogsci17/data/Mouselab_data.mat
%%
clearvars -except data data_by_sub version_nr
nr_blocks = 2;
nr_trialsPerBlock = 10;
nr_gambles = 7;
nr_outcomes = 4;

versions = {'column_order','columns_all_equal','column_EV'};
% version_nr = 2;

figdir = '~/Dropbox/mouselab_cogsci17/figures/';
SAVE = true;

nr_subjects = length(data.decision_problems);

isTTB_hDisp_hStak = zeros(nr_subjects,2);
isRandom_hDisp_hStak = zeros(nr_subjects,2);
isSATTB_hDisp_hStak = zeros(nr_subjects,2);
isTTB_hDisp_lStak = zeros(nr_subjects,2);
isRandom_hDisp_lStak = zeros(nr_subjects,2);
isSATTB_hDisp_lStak = zeros(nr_subjects,2);
isTTB_lDisp_hStak = zeros(nr_subjects,2);
isRandom_lDisp_hStak = zeros(nr_subjects,2);
isSATTB_lDisp_hStak = zeros(nr_subjects,2);
isTTB_lDisp_lStak = zeros(nr_subjects,2);
isRandom_lDisp_lStak = zeros(nr_subjects,2);
isSATTB_lDisp_lStak = zeros(nr_subjects,2);
isWADD_hDisp_hStak = zeros(nr_subjects,2);
isWADD_hDisp_lStak = zeros(nr_subjects,2);
isWADD_lDisp_hStak = zeros(nr_subjects,2);
isWADD_lDisp_lStak = zeros(nr_subjects,2);
isSAT_hDisp_hStak = zeros(nr_subjects,2);
isSAT_hDisp_lStak = zeros(nr_subjects,2);
isSAT_lDisp_hStak = zeros(nr_subjects,2);
isSAT_lDisp_lStak = zeros(nr_subjects,2);
isTTB2_hDisp_hStak = zeros(nr_subjects,2);
isTTB2_hDisp_lStak = zeros(nr_subjects,2);
isTTB2_lDisp_hStak = zeros(nr_subjects,2);
isTTB2_lDisp_lStak = zeros(nr_subjects,2);

accountedFor = false(nr_subjects,nr_blocks*nr_trialsPerBlock);

TTB_seq = [ones(1,nr_gambles),zeros(1,nr_gambles*(nr_outcomes-1))];
Random_seq = zeros(1,nr_gambles*nr_outcomes);

seq11=0;seq10=0;seq01=0;seq00=0;
for s = 1:nr_subjects
    tt = 0;
    for b = 1:nr_blocks
        for t = 1:nr_trialsPerBlock
            tt = tt+1;
            dp = data.decision_problems{s}{b}{t};
            probabilities = [dp.probabilities{:}];
            reveal_order = reshape(cell2mat([dp.reveal_order{:}]),nr_gambles,nr_outcomes)';
            [prb,ix] = sort(probabilities(:),'descend');
            isMostProb = prb==max(prb);
            reveal_order = reveal_order(ix,:);% most probable row first, etc.
            [~,ix] = sort(min(reveal_order));
            reveal_order = reveal_order(:,ix);%first selected column first, etc.
            
            rewards = repmat(cell2mat(dp.mu),nr_outcomes,1);
            EV = cell2mat(dp.mu);
            for o = 1:nr_outcomes
                for g = 1:nr_gambles
                    payoffs(o,g) = str2num(dp.payoffs{o}{g});
                end
            end
            nr_samples = sum(reveal_order(:)>0);
            
            if version_nr == 1
                index = reshape(1:nr_gambles*nr_outcomes,nr_gambles,nr_outcomes)';
            elseif version_nr == 2
                index = repmat([1:nr_outcomes]',1,nr_gambles);
            elseif version_nr == 3
                index = repmat([1:nr_gambles:nr_gambles*nr_outcomes]',1,nr_gambles);
            end
            index_rows = repmat([1:nr_outcomes]',1,nr_gambles);
            index_columns = repmat([1:nr_gambles],nr_outcomes,1);

            sampled_same_outcome = 0;
            sampled_same_gamble = 0;
            sampled_most_probable = [];
            sampled_most_probable1 = [];
            sampled_2most_probable = [];
            sequence = zeros(1,numel(reveal_order));
            for i = 1:nr_samples
                sequence(i) = index(find(reveal_order==i));
                if version_nr == 3
                    [o, g] = find(reveal_order==i);
                    rewards(o,g) = payoffs(o,g);
                    EV = mean(rewards,1);
                    EV_sort = sort(unique(EV),'descend');
                    EV_rank = 1;
                    for j = 1:length(EV_sort)
                        col = find(EV==EV_sort(j));
                        for k = 1:nr_outcomes
                            index(k,col) = nr_gambles*(k-1) + EV_rank;
                        end
                        EV_rank = EV_rank+1;
                    end
                end
                if i>1 && index_columns(find(reveal_order==i))==index_columns(find(reveal_order==i-1))
                    sampled_same_gamble = sampled_same_gamble + 1;
                elseif i>1 && index_rows(find(reveal_order==i))==index_rows(find(reveal_order==i-1))
                    sampled_same_outcome = sampled_same_outcome + 1;
                end
                if i>1 && index_rows(find(reveal_order==i))==1
                    sampled_most_probable(i-1) = true;
                elseif i>1
                    sampled_most_probable(i-1) = false;
                end
                [rr,cc] = find(reveal_order==i);
                if any(rr==find(isMostProb))
                    sampled_most_probable1(i) = true;
                else
                    sampled_most_probable1(i) = false;
                end
                if rr<=2
                    sampled_2most_probable(i) = true;
                else
                    sampled_2most_probable(i) = false;
                end
            end
            nr_gambles_selected = sum(sum(reveal_order~=0)>0);
            
% sampled_same_outcome,nr_gambles_selected,sampled_same_gamble
            
            isHighDispersion = any(probabilities>=0.85);
            isHighStakes = cell2mat(dp.mu(1))>1;
            if isHighDispersion && isHighStakes
                seq11=seq11+1;
                hDisp_hStak(seq11,:) = sequence;
                if version_nr == 2 && all(sampled_most_probable1) && nr_samples==nr_gambles %all(sequence==TTB_seq)
                    isTTB_hDisp_hStak(s,1) = isTTB_hDisp_hStak(s,1) + 1;
                    accountedFor(s,tt) = true;
                elseif version_nr == 2 %&& ~all(sequence==TTB_seq)
                    isTTB_hDisp_hStak(s,2) = isTTB_hDisp_hStak(s,2) + 1;
                end
                if all(sequence==Random_seq)
                    isRandom_hDisp_hStak(s,1) = isRandom_hDisp_hStak(s,1) + 1;
                    accountedFor(s,tt) = true;
                elseif ~all(sequence==Random_seq)
                    isRandom_hDisp_hStak(s,2) = isRandom_hDisp_hStak(s,2) + 1;
                end
%                 if version_nr==2 && sum(sequence(1:nr_gambles)==TTB_seq(1:nr_gambles))>0 && sum(sequence(1:nr_gambles)==TTB_seq(1:nr_gambles))<nr_gambles && all(sequence(nr_gambles+1:end)==TTB_seq(nr_gambles+1:end))
                if version_nr==2 && all(sampled_most_probable1) && nr_gambles_selected<nr_gambles
                    isSATTB_hDisp_hStak(s,1) = isSATTB_hDisp_hStak(s,1) + 1;
                    accountedFor(s,tt) = true;
                elseif version_nr == 2
                    isSATTB_hDisp_hStak(s,2) = isSATTB_hDisp_hStak(s,2) + 1;
                end
                if version_nr==2 && sum(logical(sequence(:)))==numel(sequence) && sampled_same_outcome==nr_gambles-1
                    isWADD_hDisp_hStak(s,1) = isWADD_hDisp_hStak(s,1) + 1;
                    accountedFor(s,tt) = true;
                elseif version_nr==2
                    isWADD_hDisp_hStak(s,2) = isWADD_hDisp_hStak(s,2) + 1;
                end
                if version_nr==2 && any(sum(reveal_order)==0) && sampled_same_outcome==(nr_gambles_selected-1) && sampled_same_gamble==(sum(sum(reveal_order)>0)*(nr_outcomes-1))
                    isSAT_hDisp_hStak(s,1) = isSAT_hDisp_hStak(s,1) + 1;
                    accountedFor(s,tt) = true;
                else
                    isSAT_hDisp_hStak(s,2) = isSAT_hDisp_hStak(s,2) + 1;
                end
                if version_nr == 2 && all(sampled_2most_probable) && nr_samples==2*nr_gambles
                    isTTB2_hDisp_hStak(s,1) = isTTB2_hDisp_hStak(s,1) + 1;
                    accountedFor(s,tt) = true;
                elseif version_nr == 2
                    isTTB2_hDisp_hStak(s,2) = isTTB2_hDisp_hStak(s,2) + 1;
                end
            elseif isHighDispersion && ~isHighStakes
                seq10=seq10+1;
                hDisp_lStak(seq10,:) = sequence;
                if version_nr == 2 && all(sampled_most_probable1)  && nr_samples==nr_gambles%all(sequence==TTB_seq)
                    isTTB_hDisp_lStak(s,1) = isTTB_hDisp_lStak(s,1) + 1;
                    accountedFor(s,tt) = true;
                elseif version_nr == 2 %&& ~all(sequence==TTB_seq)
                    isTTB_hDisp_lStak(s,2) = isTTB_hDisp_lStak(s,2) + 1;
                end
                if all(sequence==Random_seq)
                    isRandom_hDisp_lStak(s,1) = isRandom_hDisp_lStak(s,1) + 1;
                    accountedFor(s,tt) = true;
                elseif ~all(sequence==Random_seq)
                    isRandom_hDisp_lStak(s,2) = isRandom_hDisp_lStak(s,2) + 1;
                end
                if version_nr==2 && all(sampled_most_probable1) && nr_gambles_selected<nr_gambles
                    isSATTB_hDisp_lStak(s,1) = isSATTB_hDisp_lStak(s,1) + 1;
                    accountedFor(s,tt) = true;
                elseif version_nr == 2
                    isSATTB_hDisp_lStak(s,2) = isSATTB_hDisp_lStak(s,2) + 1;
                end
                if version_nr==2 && sum(logical(sequence(:)))==numel(sequence) && sampled_same_outcome==nr_gambles-1
                    isWADD_hDisp_lStak(s,1) = isWADD_hDisp_lStak(s,1) + 1;
                    accountedFor(s,tt) = true;
                elseif version_nr==2
                    isWADD_hDisp_lStak(s,2) = isWADD_hDisp_lStak(s,2) + 1;
                end
                if version_nr==2 && any(sum(reveal_order)==0) && sampled_same_outcome==(nr_gambles_selected-1) && sampled_same_gamble==(sum(sum(reveal_order)>0)*(nr_outcomes-1))
                    isSAT_hDisp_lStak(s,1) = isSAT_hDisp_lStak(s,1) + 1;
                    accountedFor(s,tt) = true;
                else
                    isSAT_hDisp_lStak(s,2) = isSAT_hDisp_lStak(s,2) + 1;
                end
                if version_nr == 2 && all(sampled_2most_probable) && nr_samples==2*nr_gambles
                    isTTB2_hDisp_lStak(s,1) = isTTB2_hDisp_lStak(s,1) + 1;
                    accountedFor(s,tt) = true;
                elseif version_nr == 2
                    isTTB2_hDisp_lStak(s,2) = isTTB2_hDisp_lStak(s,2) + 1;
                end
            elseif ~isHighDispersion && isHighStakes
                seq01=seq01+1;
                lDisp_hStak(seq01,:) = sequence;
                if version_nr == 2 && all(sampled_most_probable1)  && nr_samples==nr_gambles%all(sequence==TTB_seq)
                    isTTB_lDisp_hStak(s,1) = isTTB_lDisp_hStak(s,1) + 1;
                    accountedFor(s,tt) = true;
                elseif version_nr == 2 %&& ~all(sequence==TTB_seq)
                    isTTB_lDisp_hStak(s,2) = isTTB_lDisp_hStak(s,2) + 1;
                end
                if all(sequence==Random_seq)
                    isRandom_lDisp_hStak(s,1) = isRandom_lDisp_hStak(s,1) + 1;
                    accountedFor(s,tt) = true;
                elseif ~all(sequence==Random_seq)
                    isRandom_lDisp_hStak(s,2) = isRandom_lDisp_hStak(s,2) + 1;
                end
                if version_nr==2 && all(sampled_most_probable1) && nr_gambles_selected<nr_gambles
                    isSATTB_lDisp_hStak(s,1) = isSATTB_lDisp_hStak(s,1) + 1;
                    accountedFor(s,tt) = true;
                elseif version_nr == 2
                    isSATTB_lDisp_hStak(s,2) = isSATTB_lDisp_hStak(s,2) + 1;
                end
                if version_nr==2 && sum(logical(sequence(:)))==numel(sequence) && sampled_same_outcome==nr_gambles-1
                    isWADD_lDisp_hStak(s,1) = isWADD_lDisp_hStak(s,1) + 1;
                    accountedFor(s,tt) = true;
                elseif version_nr==2
                    isWADD_lDisp_hStak(s,2) = isWADD_lDisp_hStak(s,2) + 1;
                end
                if version_nr==2 && any(sum(reveal_order)==0) && sampled_same_outcome==(nr_gambles_selected-1) && sampled_same_gamble==(sum(sum(reveal_order)>0)*(nr_outcomes-1))
                    isSAT_lDisp_hStak(s,1) = isSAT_lDisp_hStak(s,1) + 1;
                    accountedFor(s,tt) = true;
                else
                    isSAT_lDisp_hStak(s,2) = isSAT_lDisp_hStak(s,2) + 1;
                end
                if version_nr == 2 && all(sampled_2most_probable) && nr_samples==2*nr_gambles
                    isTTB2_lDisp_hStak(s,1) = isTTB2_lDisp_hStak(s,1) + 1;
                    accountedFor(s,tt) = true;
                elseif version_nr == 2
                    isTTB2_lDisp_hStak(s,2) = isTTB2_lDisp_hStak(s,2) + 1;
                end
            elseif ~isHighDispersion && ~isHighStakes
                seq00=seq00+1;
                lDisp_lStak(seq00,:) = sequence;
                if version_nr == 2 && all(sampled_most_probable1)  && nr_samples==nr_gambles%all(sequence==TTB_seq)
                    isTTB_lDisp_lStak(s,1) = isTTB_lDisp_lStak(s,1) + 1;
                    accountedFor(s,tt) = true;
                elseif version_nr == 2 %&& ~all(sequence==TTB_seq)
                    isTTB_lDisp_lStak(s,2) = isTTB_lDisp_lStak(s,2) + 1;
                end
                if all(sequence==Random_seq)
                    isRandom_lDisp_lStak(s,1) = isRandom_lDisp_lStak(s,1) + 1;
                    accountedFor(s,tt) = true;
                elseif ~all(sequence==Random_seq)
                    isRandom_lDisp_lStak(s,2) = isRandom_lDisp_lStak(s,2) + 1;
                end
                if version_nr==2 && all(sampled_most_probable1) && nr_gambles_selected<nr_gambles
                    isSATTB_lDisp_lStak(s,1) = isSATTB_lDisp_lStak(s,1) + 1;
                    accountedFor(s,tt) = true;
                elseif version_nr == 2
                    isSATTB_lDisp_lStak(s,2) = isSATTB_lDisp_lStak(s,2) + 1;
                end
                if version_nr==2 && sum(logical(sequence(:)))==numel(sequence) && sampled_same_outcome==nr_gambles-1
                    isWADD_lDisp_lStak(s,1) = isWADD_lDisp_lStak(s,1) + 1;
                    accountedFor(s,tt) = true;
                elseif version_nr==2
                    isWADD_lDisp_lStak(s,2) = isWADD_lDisp_lStak(s,2) + 1;
                end
                if version_nr==2 && any(sum(reveal_order)==0) && sampled_same_outcome==(nr_gambles_selected-1) && sampled_same_gamble==(sum(sum(reveal_order)>0)*(nr_outcomes-1))
                    isSAT_lDisp_lStak(s,1) = isSAT_lDisp_lStak(s,1) + 1;
                    accountedFor(s,tt) = true;
                else
                    isSAT_lDisp_lStak(s,2) = isSAT_lDisp_lStak(s,2) + 1;
                end
                if version_nr == 2 && all(sampled_2most_probable) && nr_samples==2*nr_gambles
                    isTTB2_lDisp_lStak(s,1) = isTTB2_lDisp_lStak(s,1) + 1;
                    accountedFor(s,tt) = true;
                elseif version_nr == 2
                    isTTB2_lDisp_lStak(s,2) = isTTB2_lDisp_lStak(s,2) + 1;
                end
            end
        end
    end
end

%%
% 
% % isTTB_hDisp_hStak = isTTB_hDisp_hStak(:,1)./sum(isTTB_hDisp_hStak,2);
% % isRandom_hDisp_hStak = isRandom_hDisp_hStak(:,1)./sum(isRandom_hDisp_hStak,2);
% % isTTB_hDisp_lStak = isTTB_hDisp_lStak(:,1)./sum(isTTB_hDisp_lStak,2);
% % isRandom_hDisp_lStak = isRandom_hDisp_lStak(:,1)./sum(isRandom_hDisp_lStak,2);
% % isTTB_lDisp_hStak = isTTB_lDisp_hStak(:,1)./sum(isTTB_lDisp_hStak,2);
% % isRandom_lDisp_hStak = isRandom_lDisp_hStak(:,1)./sum(isRandom_lDisp_hStak,2);
% % isTTB_lDisp_lStak = isTTB_lDisp_lStak(:,1)./sum(isTTB_lDisp_lStak,2);
% % isRandom_lDisp_lStak = isRandom_lDisp_lStak(:,1)./sum(isRandom_lDisp_lStak,2);
% % mean(isTTB_hDisp_hStak)
% % mean(isRandom_hDisp_hStak)
% % mean(isTTB_hDisp_lStak)
% % mean(isRandom_hDisp_lStak)
% % mean(isTTB_lDisp_hStak)
% % mean(isRandom_lDisp_hStak)
% % mean(isTTB_lDisp_lStak)
% % mean(isRandom_lDisp_lStak)
% % mean(isTTB_hDisp_hStak(:,1)./sum(isTTB_hDisp_hStak,2))
% % mean(isRandom_hDisp_hStak(:,1)./sum(isRandom_hDisp_hStak,2))
% % mean(isTTB_hDisp_lStak(:,1)./sum(isTTB_hDisp_lStak,2))
% % mean(isRandom_hDisp_lStak(:,1)./sum(isRandom_hDisp_lStak,2))
% % mean(isTTB_lDisp_hStak(:,1)./sum(isTTB_lDisp_hStak,2))
% % mean(isRandom_lDisp_hStak(:,1)./sum(isRandom_lDisp_hStak,2))
% % mean(isTTB_lDisp_lStak(:,1)./sum(isTTB_lDisp_lStak,2))
% % mean(isRandom_lDisp_lStak(:,1)./sum(isRandom_lDisp_lStak,2))
% 
% y = sum(isTTB_hDisp_hStak(:,1))+sum(isTTB_hDisp_lStak(:,1));
% y1 = [ones(y,1);zeros(1000-y,1)];
% y = sum(isTTB_lDisp_hStak(:,1))+sum(isTTB_lDisp_lStak(:,1));
% y2 = [ones(y,1);zeros(1000-y,1)];
% [p,chi2,df,cohens_w] = chi2test({y1,y2})
% 
% y = sum(isTTB2_hDisp_hStak(:,1))+sum(isTTB2_hDisp_lStak(:,1));
% y1 = [ones(y,1);zeros(1000-y,1)];
% y = sum(isTTB2_lDisp_hStak(:,1))+sum(isTTB2_lDisp_lStak(:,1));
% y2 = [ones(y,1);zeros(1000-y,1)];
% [p,chi2,df,cohens_w] = chi2test({y1,y2})
% 
% y = sum(isRandom_hDisp_hStak(:,1))+sum(isRandom_lDisp_hStak(:,1));
% y1 = [ones(y,1);zeros(1000-y,1)];
% y = sum(isRandom_hDisp_lStak(:,1))+sum(isRandom_lDisp_lStak(:,1));
% y2 = [ones(y,1);zeros(1000-y,1)];
% [p,chi2,df,cohens_w] = chi2test({y1,y2})
% 
% y = sum(isSATTB_hDisp_hStak(:,1))+sum(isSATTB_lDisp_hStak(:,1));
% y1 = [ones(y,1);zeros(1000-y,1)];
% y = sum(isSATTB_hDisp_lStak(:,1))+sum(isSATTB_lDisp_lStak(:,1));
% y2 = [ones(y,1);zeros(1000-y,1)];
% [p,chi2,df,cohens_w] = chi2test({y1,y2})
% 
% y = sum(isTTB_hDisp_hStak(:,1))+sum(isTTB_lDisp_hStak(:,1));
% y1 = [ones(y,1);zeros(1000-y,1)];
% y = sum(isTTB_hDisp_lStak(:,1))+sum(isTTB_lDisp_lStak(:,1));
% y2 = [ones(y,1);zeros(1000-y,1)];
% [p,chi2,df,cohens_w] = chi2test({y1,y2})
% 
% y = sum(isSAT_hDisp_hStak(:,1))+sum(isSAT_lDisp_hStak(:,1));
% y1 = [ones(y,1);zeros(1000-y,1)];
% y = sum(isSAT_hDisp_lStak(:,1))+sum(isSAT_lDisp_lStak(:,1));
% y2 = [ones(y,1);zeros(1000-y,1)];
% [p,chi2,df,cohens_w] = chi2test({y1,y2})
% 
% y = sum(isWADD_hDisp_hStak(:,1))+sum(isWADD_lDisp_hStak(:,1));
% y1 = [ones(y,1);zeros(1000-y,1)];
% y = sum(isWADD_hDisp_lStak(:,1))+sum(isWADD_lDisp_lStak(:,1));
% y2 = [ones(y,1);zeros(1000-y,1)];
% [p,chi2,df,cohens_w] = chi2test({y1,y2})
% 
% disp('TBB')
% sum(isTTB_hDisp_hStak(:,1))
% sum(isTTB_lDisp_hStak(:,1))
% sum(isTTB_hDisp_lStak(:,1))
% sum(isTTB_lDisp_lStak(:,1))
% sum([sum(isTTB_hDisp_hStak(:,1))
% sum(isTTB_lDisp_hStak(:,1))
% sum(isTTB_hDisp_lStak(:,1))
% sum(isTTB_lDisp_lStak(:,1))])
% disp('Random')
% sum(isRandom_hDisp_hStak(:,1))
% sum(isRandom_lDisp_hStak(:,1))
% sum(isRandom_hDisp_lStak(:,1))
% sum(isRandom_lDisp_lStak(:,1))
% sum([sum(isRandom_hDisp_hStak(:,1))
% sum(isRandom_lDisp_hStak(:,1))
% sum(isRandom_hDisp_lStak(:,1))
% sum(isRandom_lDisp_lStak(:,1))])
% disp('SATTBB')
% sum(isSATTB_hDisp_hStak(:,1))
% sum(isSATTB_lDisp_hStak(:,1))
% sum(isSATTB_hDisp_lStak(:,1))
% sum(isSATTB_lDisp_lStak(:,1))
% sum([sum(isSATTB_hDisp_hStak(:,1))
% sum(isSATTB_lDisp_hStak(:,1))
% sum(isSATTB_hDisp_lStak(:,1))
% sum(isSATTB_lDisp_lStak(:,1))])
% disp('WADD')
% sum(isWADD_hDisp_hStak(:,1))
% sum(isWADD_lDisp_hStak(:,1))
% sum(isWADD_hDisp_lStak(:,1))
% sum(isWADD_lDisp_lStak(:,1))
% sum([sum(isWADD_hDisp_hStak(:,1))
% sum(isWADD_lDisp_hStak(:,1))
% sum(isWADD_hDisp_lStak(:,1))
% sum(isWADD_lDisp_lStak(:,1))])
% disp('SAT')
% sum(isSAT_hDisp_hStak(:,1))
% sum(isSAT_lDisp_hStak(:,1))
% sum(isSAT_hDisp_lStak(:,1))
% sum(isSAT_lDisp_lStak(:,1))
% sum([sum(isSAT_hDisp_hStak(:,1))
% sum(isSAT_lDisp_hStak(:,1))
% sum(isSAT_hDisp_lStak(:,1))
% sum(isSAT_lDisp_lStak(:,1))])
% disp('TBB2')
% sum(isTTB2_hDisp_hStak(:,1))
% sum(isTTB2_lDisp_hStak(:,1))
% sum(isTTB2_hDisp_lStak(:,1))
% sum(isTTB2_lDisp_lStak(:,1))
% sum([sum(isTTB2_hDisp_hStak(:,1))
% sum(isTTB2_lDisp_hStak(:,1))
% sum(isTTB2_hDisp_lStak(:,1))
% sum(isTTB2_lDisp_lStak(:,1))])
% disp('all')
% mean(accountedFor(:))

%%
temp = unique(hDisp_hStak,'rows');
for i = 1:size(temp,1)
    freq = 0;
    for j = 1:size(hDisp_hStak,1)
        if all(temp(i,:)==hDisp_hStak(j,:))
            freq = freq+1;
        end
    end
    hDisp_hStak_freq(i) = freq;
end
[hDisp_hStak_freq,ix] = sort(hDisp_hStak_freq,2,'descend');
hDisp_hStak = temp(ix,:);
hDisp_hStak_freq2 = [];
for i = 1:length(hDisp_hStak_freq)
    hDisp_hStak_freq2 = [hDisp_hStak_freq2,i*ones(1,hDisp_hStak_freq(i))];
end
temp = unique(hDisp_lStak,'rows');
for i = 1:size(temp,1)
    freq = 0;
    for j = 1:size(hDisp_lStak,1)
        if all(temp(i,:)==hDisp_lStak(j,:))
            freq = freq+1;
        end
    end
    hDisp_lStak_freq(i) = freq;
end
[hDisp_lStak_freq,ix] = sort(hDisp_lStak_freq,2,'descend');
hDisp_lStak = temp(ix,:);
hDisp_lStak_freq2 = [];
for i = 1:length(hDisp_lStak_freq)
    hDisp_lStak_freq2 = [hDisp_lStak_freq2,i*ones(1,hDisp_lStak_freq(i))];
end
temp = unique(lDisp_hStak,'rows');
lDisp_hStak_freq2 = [];
for i = 1:size(temp,1)
    freq = 0;
    for j = 1:size(lDisp_hStak,1)
        if all(temp(i,:)==lDisp_hStak(j,:))
            freq = freq+1;
            lDisp_hStak_freq2 = [lDisp_hStak_freq2,i];
        end
    end
    lDisp_hStak_freq(i) = freq;
end
[lDisp_hStak_freq,ix] = sort(lDisp_hStak_freq,2,'descend');
lDisp_hStak = temp(ix,:);
lDisp_hStak_freq2 = [];
for i = 1:length(lDisp_hStak_freq)
    lDisp_hStak_freq2 = [lDisp_hStak_freq2,i*ones(1,lDisp_hStak_freq(i))];
end
temp = unique(lDisp_lStak,'rows');
lDisp_lStak_freq2 = [];
for i = 1:size(temp,1)
    freq = 0;
    for j = 1:size(lDisp_lStak,1)
        if all(temp(i,:)==lDisp_lStak(j,:))
            freq = freq+1;
            lDisp_lStak_freq2 = [lDisp_lStak_freq2,i];
        end
    end
    lDisp_lStak_freq(i) = freq;
end
[lDisp_lStak_freq,ix] = sort(lDisp_lStak_freq,2,'descend');
lDisp_lStak = temp(ix,:);
lDisp_lStak_freq2 = [];
for i = 1:length(lDisp_lStak_freq)
    lDisp_lStak_freq2 = [lDisp_lStak_freq2,i*ones(1,lDisp_lStak_freq(i))];
end

%%

figure; hold on;
title('high dispersion, high stakes')
hist(hDisp_hStak_freq2,size(hDisp_hStak,1))
str=[];str2=[];
for i = 1:20
    str = [str,'num2str(hDisp_hStak(',num2str(i),',:)),'];
    str2 = [str2,'num2str(hDisp_hStak_freq(',num2str(i),')),'];
end
eval(['text(50,.6*hDisp_hStak_freq(1),{',str(1:end-1),'})'])
eval(['text(10,.6*hDisp_hStak_freq(1),{',str2(1:end-1),'})'])
if SAVE,saveas(gcf,[figdir,versions{version_nr},'_hDisp_hStak'],'png');end

figure; hold on;
title('high dispersion, low stakes')
hist(hDisp_lStak_freq2,size(hDisp_lStak,1))
str=[];str2=[];
for i = 1:20
    str = [str,'num2str(hDisp_lStak(',num2str(i),',:)),'];
    str2 = [str2,'num2str(hDisp_lStak_freq(',num2str(i),')),'];
end
eval(['text(50,.6*hDisp_lStak_freq(1),{',str(1:end-1),'})'])
eval(['text(10,.6*hDisp_lStak_freq(1),{',str2(1:end-1),'})'])
if SAVE,saveas(gcf,[figdir,versions{version_nr},'_hDisp_lStak'],'png');end

figure; hold on;
title('low dispersion, high stakes')
hist(lDisp_hStak_freq2,size(lDisp_hStak,1))
str=[];str2=[];
for i = 1:20
    str = [str,'num2str(lDisp_hStak(',num2str(i),',:)),'];
    str2 = [str2,'num2str(lDisp_hStak_freq(',num2str(i),')),'];
end
eval(['text(50,.6*lDisp_hStak_freq(1),{',str(1:end-1),'})'])
eval(['text(10,.6*lDisp_hStak_freq(1),{',str2(1:end-1),'})'])
if SAVE,saveas(gcf,[figdir,versions{version_nr},'_lDisp_hStak'],'png');end

figure; hold on;
title('low dispersion, low stakes')
hist(lDisp_lStak_freq2,size(lDisp_lStak,1))
str=[];str2=[];
for i = 1:20
    str = [str,'num2str(lDisp_lStak(',num2str(i),',:)),'];
    str2 = [str2,'num2str(lDisp_lStak_freq(',num2str(i),')),'];
end
eval(['text(50,.6*lDisp_lStak_freq(1),{',str(1:end-1),'})'])
eval(['text(10,.6*lDisp_lStak_freq(1),{',str2(1:end-1),'})'])
if SAVE,saveas(gcf,[figdir,versions{version_nr},'_lDisp_lStak'],'png');end

% plot 1: frequency of each type of sequence
% plot 2: label each cell according to most frequent,2nd,3rd,...
% plot 3: histogram in each cell showing frequency of 1st,2nd,3rd,...choice