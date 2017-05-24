function data=importData(filename,moments,delay_distributions)

%read file as text
text=fileread(filename);

if not(exist('moments','var'))
    %moments of the delay distributions
    load moments
    %rows: delay distribution type (bus_order +1)
    %columns: mean, std, skewness, kurtosis
end

if not(exist('delay_distributions','var'))
    load delay_distributions
end

%For each subject there is one row. So break up the text into rows.
rows=regexp(text,'\n','split');
rows=rows(1:end);
%the first row contains the variable names used by MTurk (e.g. workerid)
var_names=regexp(rows{1},'\t','split');
%each of the subsequent rows corresponds to one subject
nr_subjects=size(rows,2)-1;
%for each subject: read in all the variables that were saved as name-value
%pairs
for s=1:nr_subjects
   %1. read MTurk meta-data
   values_sub=regexp(rows{s+1},'\t','split');
   for v=1:length(var_names)
       if strcmp(var_names{v},'answers[question_id answer_value]')
           break;
       end
       eval(['data.',var_names{v},'{',int2str(s),'}=''',values_sub{v},'''']);
   end
   eval(['data.',strtrim(var_names{end}),'{',int2str(s),'}=''',strtrim(values_sub{end}),'''']);
   
   %2. read in the subjects' answers as strings
   temp_names=regexp(rows{s+1},'[A-Z,a-z,0-9,\_][A-Z,a-z,0-9,\_]*=','match');
   temp_values=regexp(rows{s+1},'[A-Z,a-z,0-9,\_][A-Z,a-z,0-9,\_]*=','split');
   for v=1:length(temp_names)
       if strcmp(temp_names{v}(1:end-1),'HITId')
           continue
       end
       temp_values{v+1}=strrep(temp_values{v+1},'''',''''';');
       eval(['data.',temp_names{v}(1:end-1),'_str{s}=''',strtrim(temp_values{v+1}),''';'])
   end
   
   %3. extract numeric data from subjects answers
   eval(['data.incentive_order(:,s)=',data.incentive_order_str{s},''';']);
   eval(['data.bus_order(:,s)=',data.bus_order_str{s},''';']);
   eval(['data.bonus(s)=',data.bonus_str{s},';']);
   eval(['data.subjective_time_cost(:,s)=',data.check2_str{s},';']);
   eval(['data.subjective_error_cost(:,s)=',data.check3_str{s},';']);
   
   %extract the responses the subject gave in each block
   nr_blocks=length(data.bus_order(:,s));
   data.nr_blocks=nr_blocks;
   for block_nr=1:nr_blocks
       eval(['block_str=data.block',int2str(block_nr),'_str{s}']);
       
       %extract response variables and their values
       block_str=strrep(block_str,'null','NaN');
       temp_variables=regexp(block_str,'[a-z\_A-Z]*(?=:)','match');
       temp_values=regexp(block_str,'(?<=\:)[{NaN},0-9,\-,\.,\,,\[,\]]*(\]|\d)','match');
       %enter variables and values into the block's data structure
       for v=1:length(temp_variables)
           eval(['data.block',int2str(block_nr),'.',temp_variables{v},'(:,s)=',temp_values{v},''';']);
       end
       nr_trials=size(data.(['block',int2str(block_nr)]).query_times_shuffled,1);
       eval(['data.block',int2str(block_nr),'.cost_per_unit_error(:,s)=repmat(data.block',int2str(block_nr),'.error_cost(1,s),[nr_trials,1]);']);
       eval(['data.block',int2str(block_nr),'.cost_per_sec(:,s)=repmat(data.block',int2str(block_nr),'.time_cost(1,s),[nr_trials,1]);']);
       data.(['block',int2str(block_nr)]).delay_distribution(:,s)=repmat(data.bus_order(block_nr,s),[nr_trials,1])+1;
       
       data.(['block',int2str(block_nr)]).normalized_shuffled_QT(:,s)=data.(['block',int2str(block_nr)]).query_times_shuffled(:,s)/moments(data.bus_order(block_nr,s)+1,2);
       data.(['block',int2str(block_nr)]).normalized_predictions(:,s)=data.(['block',int2str(block_nr)]).predictions(:,s)/moments(data.bus_order(block_nr,s)+1,2);
       
       %The probability of having missed the bus is the probability that
       %the bus arrived before the query time.
       data.(['block',int2str(block_nr)]).p_missed_bus(:,s)=probLessThan(delay_distributions{data.bus_order(block_nr,s)+1},data.(['block',int2str(block_nr)]).query_times_shuffled(:,s));
       data.(['block',int2str(block_nr)]).p_missed_bus_rounded(:,s)=round(10*(data.(['block',int2str(block_nr)]).p_missed_bus(:,s)))/10;
       
       data.(['block',int2str(block_nr)]).predicted_departure(:,s)=data.(['block',int2str(block_nr)]).predictions(:,s)+data.(['block',int2str(block_nr)]).query_times_shuffled(:,s);
       data.(['block',int2str(block_nr)]).normalized_predicted_departure(:,s)=data.(['block',int2str(block_nr)]).predicted_departure(:,s)/moments(data.bus_order(block_nr,s)+1,2);
       
       %data.(['block',int2str(block_nr)]).predicted_p_missed_bus(:,s)=normcdf(data.(['block',int2str(block_nr)]).normalized_predicted_departure(:,s));
       %data.(['block',int2str(block_nr)]).predicted_p_missed_bus_rounded(:,s)=round(10*normcdf(data.(['block',int2str(block_nr)]).normalized_predicted_departure(:,s)))/10;
       data.(['block',int2str(block_nr)]).predicted_p_missed_bus(:,s)=probLessThan(delay_distributions{data.bus_order(block_nr,s)+1},data.(['block',int2str(block_nr)]).predicted_departure(:,s));
       data.(['block',int2str(block_nr)]).predicted_p_missed_bus_rounded(:,s)=round(10*data.(['block',int2str(block_nr)]).predicted_p_missed_bus(:,s))/10;      
   end
   
   %concatenate the blocks: organize the data by variable and subject
   %rather than by block
   response_variables=fields(data.block1);
   for v=1:length(response_variables)
       data.(response_variables{v})=[data.block1.(response_variables{v});...
           data.block2.(response_variables{v}); data.block3.(response_variables{v});
           data.block4.(response_variables{v})];
   end
   
   
end

data.interval=20*ones(nr_trials,nr_subjects);
data.subject_nr=repmat(1:size(data.predicted_departure,2),[size(data.predicted_departure,1),1]);
data.previous_DT=[data.practice_truths(5,:);data.true_values_shuffled(1:end-1,:)];

data.examples_by_block(:,4)=[-0.7682,11.7070,-2.9390,5.9121,5.5579,-2.6188,-4.7185,-3.7777,17.6730,-1.5678]';
data.examples_by_block(:,3)=[-2.9139,-0.5261,1.4471,6.1287,0.8830,-1.6450,15.6336,-1.3616,6.7901,-0.4730]';
data.examples_by_block(:,2)=[-2.4568,3.5738,1.6174,-0.4880,-0.1115,11.2903,1.5992,4.9384,2.9069,-0.0148]';
data.examples_by_block(:,1)=[ 1.9464,7.0256,1.9309, 3.0492,0.1505,3.7596,1.6590,0.7017,1.8277,0.8911]';

data.true_departure_time=data.true_values_shuffled+data.query_times_shuffled;
data.distribution_types=[repmat(data.bus_order(1,:)+1,[20,1]);...
                 repmat(data.bus_order(2,:)+1,[20,1]);...
                 repmat(data.bus_order(3,:)+1,[20,1]);...
                 repmat(data.bus_order(4,:)+1,[20,1])];
end