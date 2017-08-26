hs = ones(numel(costs),2);
ps = ones(numel(costs),2);
num_costs = numel(costs);
num_costs = 9;
rep_sim = size(results.samples(c,2,:),3);
all_bo = zeros(num_costs*rep_sim,1);
all_mg = zeros(num_costs*rep_sim,1);
for c=1:num_costs
    %Compare BSARSA and BO
    bs = squeeze(results.samples(c,2,:));
    
    bo = squeeze(results.samples(c,5,:));
    all_bo((c-1)*rep_sim+1:c*rep_sim) = bo;
    
    mg = squeeze(results.samples(c,4,:));
    all_mg((c-1)*rep_sim+1:c*rep_sim) = mg;
    
    [h1,p1,ci1,stats1] = ttest2(bs,bo);
    [h2,p2,ci2,stats2] = ttest2(mg,bo);
    
    hs(c,1) = h1;
    hs(c,2) = h2;
    ps(c,1) = p1;
    ps(c,2) = p2;
end
[h3,p3,ci3,stats3] = ttest2(all_mg,all_bo)
    