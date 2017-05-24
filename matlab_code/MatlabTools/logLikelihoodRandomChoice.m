function ll=logLikelihoodRandomChoice(data,p)
%data: cell of structs with field choices
%choices: 1 or different from 1
%p: probability of choice==1

nr_datasets=numel(data);

ll=0;
for d=1:nr_datasets
    choice_probabilities=p*(data{d}.choices==1)+(1-p)*(data{d}.choices~=1);
    ll=ll+sum(log(choice_probabilities));
end

end