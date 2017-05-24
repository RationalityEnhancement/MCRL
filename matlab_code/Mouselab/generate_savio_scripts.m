%costs=[0.01,0.05,0.10,0.20,0.40,0.80,1.60,2.00,2.40,2.80,3.20,6.40,12.80];
costs=[0.01,0.40,0.80,1.60,2.80];
nr_initial_values=24;
continue_previous_run=false;

evaluate_BSARSA=false;
evaluate_full_observation_policy=false;

try_new_initializations=true;

if try_new_initializations
    mu0(:,1)=[1;1;1];
    mu0(:,2)=[0;0;0];
    mu0(:,3)=[0;0;1];
    mu0(:,4)=[0;1;0];
    mu0(:,5)=[1;0;0];
    mu0(:,6)=[1;1;0];
    mu0(:,7)=[1;0;1];
    mu0(:,8)=[0;1;1];
    mu0(:,9)=[0.5;0.5;0.5];
    
    sigmas=0.1:0.1:0.3;
    
    mu_ind=1:size(mu0,2);
    sigma_ind=1:numel(sigmas);
    
    [M,S]=meshgrid(mu_ind,sigma_ind);
    untried_initializations = union(find(S(:)>1),find(M==9));
    nr_initial_values=numel(untried_initializations);
    
    for init=1:nr_initial_values
        for c=1:numel(costs)
            
            script=['solve_MouselabMDP_SAVIO(',num2str(costs(c)),',',...
                int2str(init),',',int2str(continue_previous_run),');'];
            script_name=['solve_MouselabMDP_SAVIO_',int2str(c),'_',int2str(init),'.m'];
            
            fid=fopen(script_name,'w');
            fwrite(fid,script)
            fclose(fid)
            
            unix('cp ../savio_template.sh submit_job.sh')
            fid=fopen('submit_job.sh','a');
            fprintf(fid, ['\n','matlab -nodisplay -nodesktop -r "run ',...
                '/global/home/users/flieder/matlab_code/',script_name,'"']);
            fclose(fid);
            
            complete=unix('sbatch submit_job.sh');
    
        end
    end
end

if evaluate_BSARSA
    for init=1:nr_initial_values
        for c=1:numel(costs)
            
            script=['solve_MouselabMDP_SAVIO(',num2str(costs(c)),',',...
                int2str(init),',',int2str(continue_previous_run),');'];
            script_name=['solve_MouselabMDP_SAVIO_',int2str(c),'_',int2str(init),'.m'];
            
            fid=fopen(script_name,'w');
            fwrite(fid,script)
            fclose(fid)
            
            unix('cp ../savio_template.sh submit_job.sh')
            fid=fopen('submit_job.sh','a');
            fprintf(fid, ['\n','matlab -nodisplay -nodesktop -r "run ',...
                '/global/home/users/flieder/matlab_code/',script_name,'"']);
            fclose(fid);
            
            complete=unix('sbatch submit_job.sh')
        end
    end
end

for c=1:numel(costs)
    
    if evaluate_full_observation_policy
        script=['evaluateFullObservationPolicy(',num2str(costs(c)),');'];
        script_name=['observe_everything_SAVIO_',int2str(c),'.m'];
        
        fid=fopen(script_name,'w');
        fwrite(fid,script)
        fclose(fid)
        
        unix('cp ../savio_template.sh submit_job.sh')
        fid=fopen('submit_job.sh','a');
        fprintf(fid, ['\n','matlab -nodisplay -nodesktop -r "run ',...
            '/global/home/users/flieder/matlab_code/',script_name,'"']);
        fclose(fid);
        
        complete=unix('sbatch submit_job.sh')
    end
end