function  slurmTerminal(mem,cpus)
% [] = slurmTerminal(mem,cpus)
%------------------------------------------------------------------------------
% Start up a terminal on a machine in the slurm queue. Good for using cluster
% resources / prototyping
%------------------------------------------------------------------------------
% INPUT:
% <mem>:  - number indicating the amount of memory to request (in gigs)
%
% <cpus>: - number of cpus to request 
%------------------------------------------------------------------------------
% DES

if notDefined('mem'), mem = 4; end
mem = sprintf('%dGB',mem);

if notDefined('cpus'), cpus = 2; end

system(sprintf('salloc -p all -c %d --mem=%s',cpus,mem));
