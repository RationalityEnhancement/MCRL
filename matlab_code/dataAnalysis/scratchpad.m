qw=delays;
qw2=info_cost;
for j=[.01,1,2.5]
delays1=[];
delays2=[];
delays3=[];
for i=1:length(qw)
    if qw2(i)==j
        d=str2num(qw{i});
        delays1=[delays1,d(1)];
        delays2=[delays2,d(2)];
        delays3=[delays3,d(3)];
    end
end
[mean(delays1), mean(delays2), mean(delays3)]
end