function []=submitJob(job,filename,matlab_options,resource_requirements,long)

numbers=regexp(job,'[0-9]','match');
numbers=int2str(mod(str2num([numbers{:}]),1e9));
job=['rng(',numbers,'); ',job];

if not(exist('matlab_options','var'))
    matlab_options='';
end

if not(exist('resource_requirements','var'))
    resource_requirements='';
end

if not(exist('long','var'))
    long=0;
end

f_id=fopen(filename,'w');
fwrite(f_id,job);
fclose(f_id);

directory=pwd();
if long==2
    unix(['bsub -W 120:00 ',resource_requirements,' "matlab -singleCompThread -nodisplay ',matlab_options,' < ', pwd(),'/', filename,'"'])
elseif long==1
    unix(['bsub -W 35:59 ',resource_requirements,' "matlab -singleCompThread -nodisplay ',matlab_options,' < ', pwd(),'/', filename,'"'])
else
    unix(['bsub -W 7:59 ',resource_requirements,' "matlab -singleCompThread -nodisplay ',matlab_options,' < ', pwd(),'/', filename,'"'])
end

end