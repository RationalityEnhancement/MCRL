function [p,chi2,df,cohens_w] = chi2test(data)
%[p,chi2,df,cohens_w] = chi2test(data)
%Performs the Pearson's Chi^2 test. data is a cell array and each cell
%is interpreted as the responses of one group. This function tests the null hypothesis that
%the the data are distributed identically in all groups.

nr_groups=numel(data);

%determine the distribution under the null hypothesis
all_data=[];
for g=1:nr_groups
    all_data=[all_data;data{g}(:)];
end

values=unique(all_data);
nr_values=numel(values);

for v=1:nr_values
    relative_frequency(v)=mean(all_data==values(v));
    for g=1:nr_groups
        expected_frequency(v,g)=relative_frequency(v)*numel(data{g}(:));
        observed_frequency(v,g)=sum(data{g}(:)==values(v));
        discrepancy(v,g)=(observed_frequency(v,g)-expected_frequency(v,g))^2/expected_frequency(v,g);
    end
end

chi2=sum(discrepancy(:));
cohens_w=sqrt(chi2);
nr_cells=nr_groups*nr_values;
nr_parameters=nr_values-1;
df=(nr_values-1)*(nr_groups-1); %see Wikipedia article on Pearson's Chi2 Test, Section "Test of Independence".
p=1-chi2cdf(chi2,df);

end