for j=1:10000 % each sequence of observations
    nheads = 1;
    total = 2;
    for i=1:1000 %observe for 1000 steps    
        flip = rand;
        pheads = nheads/total;
        heads = flip <= pheads;
        if heads
            nheads = nheads + 1;
        end
        total = total + 1;
        if mod(i,10) == 0
            h(j,i/10) = nheads;
%             er(j,i/10) = max(nheads,total-nheads)-min(nheads,total-nheads);
%             er(j,i/10) = (max(nheads,total-nheads)-min(nheads,total-nheads))/total;
            er(j,i/10) = max(nheads,total-nheads)/total;
        end
    end
end

for i=1:100
    mu(i) = mean(er(:,i));
    v(i) = var(er(:,i));
end
    

% for j=1:1000 % each sequence of observations
%     nheads = 1;
%     total = 2;
%     for i=1:100 %observe for 1000 steps    
%         flip = rand;
%         pheads = nheads/total;
%         heads = flip <= pheads;
%         if heads
%             nheads = nheads + 1;
%         end
%         total = total + 1;
%         h(j,i) = nheads;
%         er(j,i) = (max(nheads,total-nheads)-min(nheads,total-nheads))/total;
%     end
% end
% 
% for i=1:100
%     mu(i) = mean(er(:,i));
%     v(i) = var(er(:,i));
% end
%     