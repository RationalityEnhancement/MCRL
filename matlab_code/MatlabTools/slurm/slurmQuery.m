function slurmQuery(varargin);
%[] = slurmQuery(varargin);
%------------------------------------------------------------------------------
% Query the slurm queue
%------------------------------------------------------------------------------
% USES:
% slurmCancel()
%   gives information on all jobs
%
% slurmCancel(jobID):
%   cancels all jobs with id <jobID>
%
% slurmCancel(option,argument): 
%   cancels jobs based on default slurm protocol
%   (use >> !man squeue for more details 
%   options include:
%        '-n':   query job with name <argument>
%        '-j':   query job with job id <argument>
%        '-u':   query all jobs for user <argument>
%        '-p':   query all jobs on partition <argument>
%        '-qos': query all jobs with quality of service
%                <argument>.
%        '-t': : query all jobs in state <argument> (e.g. 'PENDING','RUNNING',
%                'COMPLETED','CONFIGURING','CANCELLED','FAILED', 'TIMEOUT', etc)
%
% OUTPUT:
% <outPut>: - command line string output from running the command
%
% <status>: - status of command
%------------------------------------------------------------------------------
%DES


if nargin == 0
%    comStr = 'squeue';
	slurmMap;
	return
end

if nargin == 1% DEFAULT TO QUERYING JOBS FOR A USER
%  	comStr = sprintf('squeue -u %s',varargin{1});
	if isnumeric(varargin{1}), varargin{1} = num2str(varargin{1}); end
	comStr = sprintf('squeue -j %s',varargin{1});
end

if nargin == 2

	if isempty(findstr(varargin{1},'-')), varargin{1} = ['-',varargin{1}];end
	if isnumeric(varargin{2}), varargin{2} = num2str(varargin{2}); end
	comStr = sprintf('squeue %s %s',varargin{1},varargin{2});
end

[status, outPut]=system([comStr, ' -l']);

disp(outPut);
