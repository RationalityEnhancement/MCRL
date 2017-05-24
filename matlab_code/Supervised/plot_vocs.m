states = [1,1;3,1;5,4; 8,1; 8,8; 10,1;12,12;24,1];
cost = 0.001;
for i=1:8
    st = states(i,:);
    vocs = zeros(30,1);
    for j=1:30
        vocs(j) = nvoc(j+1,st,cost);
    end
    subplot(4,2,i)
    plot(vocs);
    title(st);
end
