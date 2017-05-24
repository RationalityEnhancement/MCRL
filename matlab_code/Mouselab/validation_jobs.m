costs=[0.01,0.05,0.10,0.20,0.40,0.80,1.60,2.00,2.40,2.80,3.20,6.40,12.80];

for c=1:numel(costs)
    
   script=['validateLearnedPolicy(',int2str(c),')'];
   script_name=['validation_job_',int2str(c),'.m'];
    
   submitJob2Savio(script,script_name) 
end