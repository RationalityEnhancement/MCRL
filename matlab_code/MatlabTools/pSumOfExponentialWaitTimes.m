function [p_waiting_time,times]=pSumOfExponentialWaitTimes(lambda1,lambda2,times)

p_time1=lambda1*exp(-lambda1*times) / sum( lambda1*exp(-lambda1*times));
p_time2=lambda2*exp(-lambda2*times) / sum( lambda2*exp(-lambda2*times));

[sum_times,p_waiting_time_sum]=sumpdf(times,p_time1,1,times,p_time2,1);

min_sum_time=min(sum_times);
p_waiting_time=zeros(size(times));
p_waiting_time(times<min_sum_time)=0;

for t=1:numel(times)
    index=find(sum_times==times(t));
    
    if ~isempty(index) 
        p_waiting_time(t)=p_waiting_time_sum(index);
    end

end

p_waiting_time(end)=sum(p_waiting_time_sum(sum_times>=max(times)));

end