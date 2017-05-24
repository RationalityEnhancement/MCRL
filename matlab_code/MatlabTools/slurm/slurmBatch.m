function jobIds = slurmBatch(cmds, jobParams, matlabRunner)
% jobIds = slurmBatch(cmds, [jobParams], [matlabRunner])
%-------------------------------------------------------------------------
% Run a string of MATLAB commands using slurm. If no arguments provided
% returns a default list of jobParameters.
%-------------------------------------------------------------------------
% INPUT:
% <cmds>:         Either a single string of commands, or a
%                 cell array of strings. If a cell array, each
%                 element is run using it's own job.
%
% <jobParams>:    Parameters for the jobs:
%                .partition: the name of the partition to run on
%                .depends: a comma-separated list of job id dependencies
%                .out: file path to write stdout to
%                .err: file path to write stderr to
%                .nodes: the # of nodes this job will require
%                .qos: the quality-of-service for this job
%                .cpus: the # of CPUs required by this job
%                .nodelist: a comma-separated list of specific nodes to run on
%                .memory: the memory in megabytes that your job requires
%                .maxCompThreads: maximum number of computational threads
%
% <matlabRunner>: the script used to run matlab commands,
%                 defaults to 'matlab_background'
%
% OUTPUT:
% <jobIds>:       a vector of job IDs, the same size as cmds
%----------------------------------------------------------------------

% SET THIS TO WHERE YOU WANT SCRIPTS, LOGS, AND WHATNOT SAVED
outHome = '/global/home/users/flieder/slurm/';

if ~nargin || any(strcmp(cmds,{'params','jobParams'})) 
   jobIds = getDefaultParams;
   return
end

if ~iscell(cmds)
    cmds = {cmds};
end

if notDefined('jobParams')
    jobParams = getDefaultParams;
end

if notDefined('matlabRunner')
    matlabRunner = 'matlab_background'; % THIS SHELL SCRIPT NEEDS TO BE IN YOUR SHELL PATH
end

% DEFAULTS
if isfield(jobParams,'name') && ~isempty(jobParams.name)
   jobname = jobParams.name;
else
   jobname = 'slurmBatch';
end

if ~isfield(jobParams, 'partition') || isempty(jobParams.partition)
    jobParams.partition = 'all';
end

if ~isfield(jobParams, 'cpus') || isempty(jobParams.cpus)
    jobParams.cpus = 1;
end

if ~isfield(jobParams, 'memory') || isempty(jobParams.memory)
    %specify 1GB/CPU
    jobParams.memory = jobParams.cpus * 7500;
end

if jobParams.memory < 1000
    fprintf(sprintf('WARNING: assuming that memory provided (%1.2f) is in GB, multiplying by 1024\n',jobParams.memory));
    jobParams.memory = jobParams.memory*1024;
end

% CONSTRUCT LIST OF COMMANDS TO PASS TO SBATCH
sbatchCmds = {};
if isfield(jobParams, 'partition') && ~isempty(jobParams.partition)
    sbatchCmds{length(sbatchCmds)+1} = '-p';
    sbatchCmds{length(sbatchCmds)+1} = jobParams.partition;
end

if isfield(jobParams, 'depends') && ~isempty(jobParams.depends)
    sbatchCmds{length(sbatchCmds)+1} = '-d';
    dependencyStr = ['afterok',sprintf(':%d',jobParams.depends)];
    sbatchCmds{length(sbatchCmds)+1} = dependencyStr;
end

if isfield(jobParams, 'out') && ~isempty(jobParams.out)
    sbatchCmds{length(sbatchCmds)+1} = '-o';
    sbatchCmds{length(sbatchCmds)+1} = jobParams.out;
else
    outstr = fullfile(outHome, sprintf('%s-%s.log',jobname,datestr(now,'yyyymmddHHMMSSFFF')));
    sbatchCmds{length(sbatchCmds)+1} = '-o';
    sbatchCmds{length(sbatchCmds)+1} = outstr;
end

% HERE WE KEEP THE LOG FILE FOR LOOKUP LATER
if numel(cmds) == 1
    logFile = outstr;
end

if isfield(jobParams, 'err') && ~isempty(jobParams.err)
    sbatchCmds{length(sbatchCmds)+1} = '-e';
    sbatchCmds{length(sbatchCmds)+1} = jobParams.err;
else
    outstr = fullfile(outHome, sprintf('%s-%s.err',jobname,datestr(now,'yyyymmddHHMMSSFFF')));
    sbatchCmds{length(sbatchCmds)+1} = '-e';
    sbatchCmds{length(sbatchCmds)+1} = outstr;
end

% HERE WE KEEP THE ERROR FILE FOR LOOKUP LATER
if numel(cmds) == 1
  errorFile = outstr;
end

if isfield(jobParams, 'nodes') && ~isempty(jobParams.nodes)
    sbatchCmds{length(sbatchCmds)+1} = '-N';
    sbatchCmds{length(sbatchCmds)+1} = num2str(jobParams.nodes);
end

if isfield(jobParams, 'qos') && ~isempty(jobParams.qos)
    sbatchCmds{length(sbatchCmds)+1} = sprintf('--qos=%d', jobParams.qos);
end

if isfield(jobParams, 'cpus') && ~isempty(jobParams.cpus)
    sbatchCmds{length(sbatchCmds)+1} = '-c';
    sbatchCmds{length(sbatchCmds)+1} = num2str(jobParams.cpus);
end

if isfield(jobParams, 'nodelist') && ~isempty(jobParams.nodelist)
    sbatchCmds{length(sbatchCmds)+1} = '-w';
    sbatchCmds{length(sbatchCmds)+1} = num2str(jobParams.nodelist);
end

if isfield(jobParams, 'name') && ~isempty(jobParams.name)
    sbatchCmds{length(sbatchCmds)+1} = '-J';
    sbatchCmds{length(sbatchCmds)+1} = ['"' jobParams.name '"'];
else
    sbatchCmds{length(sbatchCmds)+1} = '-J';
    sbatchCmds{length(sbatchCmds)+1} = 'slurmBatch';
end

if isfield(jobParams, 'memory') && ~isempty(jobParams.memory)
    sbatchCmds{length(sbatchCmds)+1} = sprintf('--mem=%d', jobParams.memory);
end

if isfield(jobParams,'maxCompThreads') && ~isempty(jobParams.maxCompThreads)
    maxCompThreads = jobParams.maxCompThreads;
else
    maxCompThreads = 2*jobParams.cpus;
end

sbatchCmds{length(sbatchCmds)+1} = sprintf('--comment="%s"', cmds{1});

sbatchRunCmd = 'sbatch';
for iC = 1:length(sbatchCmds)
    sbatchRunCmd = [sbatchRunCmd ' ' sbatchCmds{iC}];
end

jobIds = ones(1, length(cmds))*-1;

for iC = 1:length(cmds)

    % WRITE OUT EXECUTABLE COMMAND IN SCRIPT
    tempOut = tempname();
    fid = fopen(tempOut, 'w');
    fprintf(fid, '#!/bin/zsh\n');
    fprintf(fid, '#SBATCH\n');

    mcmd = cmds{iC};
    mcmd = strrep(mcmd, '\n', '');

    mThreadStr = sprintf('maxNumCompThreads(%d); ', maxCompThreads);

    mcmd = ['"' mThreadStr ' ' mcmd '"'];
    fprintf(fid, '%s %s\n', matlabRunner, mcmd);
    fclose(fid);
   
    % CONSTRUCT FINAL SBATCH 
    sbatchFinalCmd = [sbatchRunCmd ' ' tempOut];

    % EXECUTE 
    [status, result] = system(sbatchFinalCmd);

    if status ~= 0
        error('sbatch.m: problem executing command: %s', result);
    end 

    % GET JOB 
    mat = regexp(result, 'job\s\d*', 'match');
    if ~isempty(mat)
         jobIds(iC) = str2num(mat{1}(4:end));
    end

    if numel(cmds) == 1
        eval(sprintf('job%d=struct;',jobIds));
        eval(sprintf('job%d.log=logFile;',jobIds));
        eval(sprintf('job%d.err=errorFile;',jobIds));
        logLookup = fullfile(outHome,'logLookup.mat');

        try
            eval(sprintf('save(''%s'',''job%d'',''-append'',''-v7.3'');',logLookup,jobIds));
        catch
            eval(sprintf('save(''%s'',''job%d'',''-v7.3'');',logLookup,jobIds))
        end
    end
end

function jobParams = getDefaultParams()
% RETURN DEFAULT JOB PARAMETERS
jobParams.cpus = 1;
jobParams.partition = 'all'; % YOU MAY NEED TO CHANGE THIS, DEPENDING ON YOUR SLURM SETUP
jobParams.memory = [];
jobParams.name = [];
jobParams.out = [];
jobParams.err = [];
jobParams.nodes = [];
jobParams.qos = [];
jobParams.depends = [];
jobParams.nodelist = [];
jobParams.maxCompThreads = [];


