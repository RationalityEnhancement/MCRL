function [p_sum,values]=pSum(p1,p2,values)

[sum_values,p_sums]=sumpdf(values,p1,1,values,p2,1);

min_sum_time=min(sum_values);
p_sum=zeros(size(values));
p_sum(values<min_sum_time)=0;

start_overlap=find(values==sum_values(1));
end_overlap=find(sum_values==values(end));

if isempty(start_overlap)
    length_overlap=0;
else
    length_overlap=end_overlap-start_overlap+1;
end

p_sum(1,1:(start_overlap-1))=0;
p_sum(1,start_overlap:end_overlap)=p_sums(1:length_overlap);
p_sum(end_overlap)=sum(p_sums(end_overlap:end));

end