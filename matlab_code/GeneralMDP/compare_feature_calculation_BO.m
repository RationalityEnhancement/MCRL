X = nlightbulb_problem.fit.features;
tic
for i=1:nr_states
        st = S(i,:);

        I = find(all(repmat(st,size(S,1),1)==S,2));
        f_obs4 = X(nr_arms*(I-1)+(1:nr_arms),1:2);
end
toc

tic
for i=1:nr_states
        st = S(i,:);
        st_m = reshape(st,2,nr_arms)';
        voc1 = zeros(nr_arms,1);
        vpi = zeros(nr_arms,1);
        for a=1:nr_arms
            voc1(a) = VOC1MultiArmBernoulli(st_m(:,1),st_m(:,2),a,co);
            vpi(a) = valueOfPerfectInformationMultiArmBernoulli(st_m(:,1),st_m(:,2),a);
        end
        f_obs = cat(2,voc1(:), vpi(:));
end
toc

tic
for i=1:nr_states
        st = S(i,:);
        
        I = find(all(repmat(st,size(S,1),1)==S,2));
        f_obs4 = X(nr_arms*(I-1)+(1:nr_arms),1:2);
        
        st_m = reshape(st,2,nr_arms)';
        voc1 = zeros(nr_arms,1);
        vpi = zeros(nr_arms,1);
        for a=1:nr_arms
            voc1(a) = VOC1MultiArmBernoulli(st_m(:,1),st_m(:,2),a,co);
            vpi(a) = valueOfPerfectInformationMultiArmBernoulli(st_m(:,1),st_m(:,2),a);
        end
        f_obs = cat(2,voc1(:), vpi(:));
        
        if not(isequal(f_obs,f_obs4))
            disp(i);
            disp('heck');
        end
end
toc