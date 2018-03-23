
    
% cd ~/Dropbox/'Accelerating Learning with PRs'/

addpath('~/Dropbox/PhD/MatlabTools/')
addpath('~/Dropbox/PhD/MatlabTools/parse_json/')

filename_metadata = '~/Dropbox/mouselab_cogsci17/data/mouselab_cogsci17_metadata.csv';

load ../data/Mouselab_data.mat

for s = [1:33,35:200]%length(data_by_sub)
    if ~isfield(data_by_sub{s},'bonus')
        bonus(s) = 3.53;
    else
        bonus(s) = data_by_sub{s}.bonus;
    end
end

text=fileread(filename_metadata);

worker_IDs=regexp(text,'[A-Z0-9]*(?=,Approved,)','match');

temp=regexp(text,'html,[A-Z0-9]*','match');
for s=1:numel(temp)
    assignment_IDs{s}=temp{s}(6:end);
end

bonus_file='';
for s=1:numel(worker_IDs)
    
    bonus_commands{s}=['./grantBonus.sh -workerid ',worker_IDs{s},...
        ' -assignment ', assignment_IDs{s}, ' -amount ', num2str(bonus(s)),...
        ' -reason "Bonus in Betting Game."'];
    bonus_file=[bonus_file, bonus_commands{s},';'];
end

filename=['payBonuses_mouselab_cogsci17.sh'];
unix(['rm ',filename])
fid = fopen(filename, 'w');
% print a title, followed by a blank line
fprintf(fid, bonus_file);
fclose(fid)
unix(['chmod u+x ',filename])
unix(['mv ',filename,' /Applications/aws-mturk-clt-1.3.1/bin/'])

