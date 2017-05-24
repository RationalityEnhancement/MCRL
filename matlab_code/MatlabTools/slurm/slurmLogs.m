function slurmLogs(jobID,logType);
%  slurmLogs(jobID,[logType]);
%------------------------------------------------------------------------
% Prints the output of log files based on jobID returned by slurmBatch.m
% <logType> is a string (either 'log' or 'err') indicating which output
% type to print. Default is 'log', dislaying the stdout of the slurm 
% job.
%
% Note: can also purge all logs with the command
% >> slurmLogs('purge')
%------------------------------------------------------------------------
% DES

% NEED TO EDIT THIS TO BE THE SAME AS slurmBatch.m 
outHome = '/global/home/users/flieder/slurm/';
logLookup = fullfile(outHome,'logLookup.mat');

if ischar(jobID)
	
	if strcmp(lower(jobID),'purge')
		ans = input('purge all log files (y/n)?','s');
		if strcmp(ans,'y')
			system(sprintf('rm -r %s*',fullfile(outHome)))
			jobNull = [];
			save(logLookup,'jobNull','-v7.3')
		end
	return
	end
end

if notDefined('logType'),logType = 'log'; end


m = matfile(logLookup);

fName = sprintf('job%d',jobID);

try
	m = m.(fName);

	switch logType
		case {'log'}
			catFile = m.log;
		case {'error','err'}
			catFile = m.err;
	end

	type(catFile);
	
catch
	error(sprintf('no logs for job: %d',jobID));
end
