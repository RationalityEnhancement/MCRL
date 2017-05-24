costs=[0.01,0.05,0.10,0.20,0.40,0.80,1.60,2.00,2.40,2.80,3.20,6.40,12.80];

for c=1:length(costs)
    job=['policySearchMouselabMDPSavio(',num2str(costs(c)),')'];
    job_name=['BO_Mouselab_c',int2str(100*c),'.m'];
    submitJob2Savio(job,job_name)
end