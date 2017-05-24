function submitJob2Savio(script,script_name)

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