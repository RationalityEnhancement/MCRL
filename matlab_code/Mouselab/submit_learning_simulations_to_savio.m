%submit Mouselab learnign simulations to Savio

nr_conditions=3;
nr_reps=20;
for r=1:nr_reps
    for c=1:nr_conditions
        job=['simulateMouselabLearningSavio(',int2str(r),',',int2str(c),',true)'];
        job_name=['Mouselab_learning_simulation_c',int2str(c),'_r',int2str(r),'_withPR.m'];
        submitJob2Savio(job,job_name)
        
        job=['simulateMouselabLearningSavio(',int2str(r),',',int2str(c),',false)'];
        job_name=['Mouselab_learning_simulation_c',int2str(c),'_r',int2str(r),'_withoutPR.m'];
        submitJob2Savio(job,job_name)
    end
end