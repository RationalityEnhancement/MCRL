function [ER_hat,result]=evaluateGeneralMDP(w,nr_arms,nr_balls)
load(['../../../results/', num2str(nr_arms),'lightbulb_fit.mat'])
reps = 500;
horizon = nr_balls+1;
reward = 0;
X = nlightbulb_problem.fit.features;
S = nlightbulb_problem.mdp.states;
co =  nlightbulb_problem.mdp.cost;
for j=1:reps
    st = S(1,:);
    I = 1;
    r = 0;
    for i=1:horizon
        %decision based on w

%         I = find(S(:, 1) == st(1) & S(:, 2) == st(2));
        I = find(all(repmat(st,size(S,1),1)==S,2));
        f_obs = X(nr_arms*(I-1)+(1:nr_arms),1:4);

        st_m = reshape(st,2,nr_arms)';
%         voc1 = zeros(nr_arms,1);
%         vpi = zeros(nr_arms,1);
%         for a=1:nr_arms
%             voc1(a) = VOC1MultiArmBernoulli(st_m(:,1),st_m(:,2),a,co);
%             vpi(a) = valueOfPerfectInformationMultiArmBernoulli(st_m(:,1),st_m(:,2),a);
%         end
%         f_obs = cat(2,voc1(:), vpi(:));
        [m,idx] = max(f_obs*w);
        flip = rand;
%                         pheads = cs(1)/(cs(1)+cs(2));
              
        if m > 0
            r = r - co;
            heads = flip <= st_m(idx,1)/sum(st_m(idx,:));
            if heads
                st_m(idx,1) = st_m(idx,1)+1;
            else
                st_m(idx,2) = st_m(idx,2)+1;
            end
            st_mt = st_m';
            st = st_mt(:)';
        else
            [m,~] = max(st_m(:,1)./sum(st_m,2));     
            r = r + m;
            break
        end
    end
    reward = reward + r;
end
ER_hat = reward/reps;
result.features={'VPI','VOC','VPI_all','cost'};
result.cost_per_click=co;
end