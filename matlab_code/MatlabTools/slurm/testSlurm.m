slurmOpts = slurmBatch;
slurmOpts.memory = 1.5;
slurmOpts.cpus = 1;
slurmOpts.partition = 'all'

jids = [];
nTests = 10;
for ii = 1:nTests
	comStr = sprintf('startup; pause(30); disp(''job %d complete'')',ii);
	slurmOpts.name = sprintf('test_%d',ii);
	slurmBatch(comStr,slurmOpts);
end