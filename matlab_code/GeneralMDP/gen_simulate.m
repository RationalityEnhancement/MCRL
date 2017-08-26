n = size(nlightbulb_problem.mdp.pi_star);
co = nlightbulb_problem.mdp.cost;
pols = zeros(n(1)-1,5);
[m0, pols(:,1)] = max(nlightbulb_problem.mdp.Q_star(1:n(1)-1,:),[],2);
[m1, pols(:,2)] = max(nlightbulb_problem.fit.Q_hat,[],2);
[m2, pols(:,3)] = max(nlightbulb_problem.bsarsa.Q_hat_BSARSA(1:n(1)-1,:),[],2);
pols(:,4) = nlightbulb_problem.fit.pi_meta(1:n(1)-1);
pols(:,5) = nlightbulb_problem.BO.pi_BO(1:n(1)-1);

reps=10000;
samples = zeros(reps,5);
for p=1:5
    reward = 0;
for j=1:reps
    st = S(1,:);
    I = 1;
    r = 0;
    for i=1:nr_balls+1
        flip = rand;

        I = find(all(repmat(st,size(S,1),1)==S,2));
        st_m = reshape(st,2,nr_arms)';
        a = pols(I,p);
%         disp(i);
%         disp(st);
%         disp(st_m);
%         disp(a);

        if a == nr_arms+1 || sum(st) == 2*nr_arms+nr_balls
%             disp('here');
            [m,idx] = max(st_m(:,1)./sum(st_m,2));

%             heads = flip <= m;
% 
%             samples(j,p) = r;
%             if heads
                r = r + m;
                samples(j,p) = r ;
%             end
            break
        elseif a < nr_arms + 1 
%             disp('instead here');
            heads = flip <= st_m(a,1)/sum(st_m(a,:));
            r = r - co;
            if heads
                st_m(a,1) = st_m(a,1)+1;
            else
                st_m(a,2) = st_m(a,2)+1;
            end
%             disp(st_m);

            st_m = st_m';
            st = st_m(:)';
            
%             disp(st);
        end
    end
    reward = reward + r;
end
    ER_hat = reward/reps;
    disp(ER_hat);
end