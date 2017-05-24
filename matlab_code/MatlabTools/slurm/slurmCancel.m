function [outPut,status] = slurmCancel(varargin);
%outPut = slurmCancel(varargin);
%------------------------------------------------------------------------------
% Cancels slurm jobs based on vararin. 
%------------------------------------------------------------------------------
% USES:
% slurmCancel()
% 
% slurmCancel(jobIDs):          - cancels all jobs with job id <jobIDs>
%
% slurmCancel(userID):          - cancels all jobs by user <userID>
%
% slurmCancel(option,argument): - cancels jobs based on default slurm protocol
%                                 (use >> !man scancel for more details) 
%                                 options include:
%                                     '-n': cancel job with name <argument>
%                                     '-u': cancel all jobs for user <argument>
% OUTPUT:
% <outPut>:                     - command line string output from running the command
%
% <status>:                     - status of command
%------------------------------------------------------------------------------
%DES


if nargin == 0
	
	username = getenv('USER'); % ONLY WORKS ON UNIX
	
    resp = input(sprintf('want to quit all slurm jobs for user %s?\n',username),'s');
    switch resp
        case {'y','yes'}
        comStr = sprintf('scancel -u %s',username);
    end
    return
end

if nargin == 1
	resp = input(sprintf('want to quit slurm jobs?\n'),'s');
	switch resp
	case {'y','yes'}
	if ischar(varargin{1})
		comStr = sprintf('scancel -u %s',varargin{1});
		[status, outPut]=system(comStr);

	elseif isnumeric(varargin{1})
		for id = 1:numel(varargin{1})
			comStr = sprintf('scancel %d',varargin{1}(id));
			[status, outPut]=system(comStr);
		end
		return
	end
	end
end

if nargin == 2
	comStr = sprintf('scancel %s %s',varargin{1},varargin{2});	
end
[status, outPut]=system(comStr);
