% filename=['../data/05302018/DataExp',int2str(experiment_nr),'JSON.csv'];
experiment_nr = '05302018';
filename=['../data/',experiment_nr,'/DataExp2JSON.csv'];
filename_metadata=['../data/',experiment_nr,'/MetaData.csv'];

addpath ../../matlab_code/MatlabTools/parse_json/
addpath ../../matlab_code/MatlabTools/

fid=fopen(filename);
nr_subjects=linecount(fid)-1;
fclose(fid);

fid=fopen(filename)
header = fgetl(fid);


for sub=1:nr_subjects
    subject_str = fgetl(fid);
    subject_str=strrep(regexprep(subject_str,'["]+','"'),'[]','[-999999999]');
    subject_str=regexprep(subject_str,'(?<=\:[0-9\.]+)"','');
    subject_str=regexprep(subject_str,'(?<=(\:\[[0-9\.]+)+)"','');
    subject_str=regexprep(subject_str,'(?<=\[[0-9]+)"','');
    temp=parse_json(subject_str(2:end-1));
    data_by_sub{sub}=temp{1};
    disp(['Loaded data from subject ',int2str(sub)])
end
fclose(fid)

data=cellOfStructs2StructOfCells(data_by_sub)
save(['../data/',experiment_nr,'/Mouselab_data_Experiment.mat'], 'data_by_sub','data')

%{
% pay bonuses

fid=fopen(filename_metadata)
text = fileread(filename_metadata);
%worker IDs
worker_IDs=regexp(text,'[A-Z0-9]*(?=,Approved,)','match');
nr_subjects=numel(worker_IDs);
%assignment IDs
temp=regexp(text,'html",[A-Z0-9]*','match');
for s=1:numel(temp)
    assignment_IDs{s}=temp{s}(7:end);
end
%}