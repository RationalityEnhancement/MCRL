sigmas = [0.39; 0.54; 0.12];

for s=1:numel(sigmas)
    for k=1:16
        [mu(k,s),sigma]=EVofMaxOfGaussians(zeros(k,1),sigmas(s)*ones(k,1));
    end
end

thresholds_low_cost  = [0;mu(:,1)];
thresholds_med_cost  = [0;mu(:,2)];
thresholds_high_cost = [0;mu(:,3)];

savejson('',thresholds_low_cost,'../experiments/exp1/static/json/thresholds_low_cost.json')
savejson('',thresholds_med_cost,'../experiments/exp1/static/json/thresholds_med_cost.json')
savejson('',thresholds_high_cost,'../experiments/exp1/static/json/thresholds_high_cost.json')

thresholds.low_cost=thresholds_low_cost
thresholds.med_cost=thresholds_med_cost
thresholds.high_cost=thresholds_high_cost
save DataAnalysis/PR_thresholds.mat thresholds