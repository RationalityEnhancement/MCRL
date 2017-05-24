costs=[0.01,0.05*power(2,0:8)];

for c=1:length(costs)
    job=['evaluateBlinkeredApproximationSavio(',num2str(costs(c)),')'];
    job_name=['evaluate_blinkered_approximation_',int2str(c),'.m'];
    
    submitJob2Savio(job,job_name)
end