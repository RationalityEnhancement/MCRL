function samples=sampleDiscreteDistributions(distributions,nr_samples)
%distributions: matrix of probability vectors, each row is one distribution

nr_distributions=size(distributions,1);
nr_values=size(distributions,2);
values=1:nr_values;

is=rand(nr_distributions,nr_samples);

cdfs=[zeros(nr_distributions,1),cumsum(distributions')'];
cdfs(:,end)=ones(nr_distributions,1);
vals=repmat(values,[nr_distributions,1]);

samples=nan(nr_distributions,nr_samples);
for s_ind=1:nr_samples
    condition_met=repmat(is(:,s_ind),[1,nr_values])<=cdfs(:,2:end) & repmat(is(:,s_ind),[1,nr_values])>=cdfs(:,1:(end-1));
    [row_ind col_ind]=find(condition_met);
    [row_inds order]=sort(row_ind); col_inds=col_ind(order);
    indices=sub2ind([nr_distributions,nr_values],row_inds,col_inds);
    
    samples(:,s_ind)=vals(indices);    
end

end