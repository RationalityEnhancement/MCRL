[minval, idx] = min(matm(:));
[aidx, bidx, cidx] = ind2sub(size(matm), idx);
matmeanm = zeros(size(mus,2),numel(sigmas));
matmeanr2 = zeros(size(mus,2),numel(sigmas));

for m=1:size(mus,2)
    for sig=1:numel(sigmas)
        matmeanm(m,sig) = mean(matm(m,sig,:));
        matmeanr2(m,sig) = mean(matr(m,sig,:));
        matstdm(m,sig) = std(matm(m,sig,:));
        matstdr2(m,sig) = std(matm(m,sig,:));
    end
end

[i_mu,i_sig] = find(matmeanm == min(matmeanm(:)));
[min_m, i_rep] = min(matm(i_mu,i_sig,:));
w = squeeze(matws(i_mu,i_sig,i_rep,:));

%% Check result
h=6;
c = 1;
costs = 0.01;
mdp=generalMDP(nr_actions,gamma,nr_features,costs(c),h); 
for s=1:nr_states
    st = nlightbulb_problem(c).mdp.states(s,:);
    st_m = reshape(st,2,nr_actions)';
    er = max( st_m(:,1) ./ sum(st_m,2));
    for a=1:nr_actions+1          
        F=fexr(st_m,a,mdp);
        Q_hat(s,a)=F'*w;
    end
end

nr_arms = size(states(1,:),2)/2;

nr_observations=sum(states,2)-2*nr_arms;
max_nr_observations=max(nr_observations); 

valid_states=find(and(nr_observations<max_nr_observations,...
    nr_observations>=0));

V_hat=max(Q_hat,[],2);
qh = Q_hat(valid_states,:);
qs = nlightbulb_problem(c).fit.Q_star(valid_states,:);
R2(c)=corr(qh(:),qs(:))^2;

fig_Q=figure();
scatter(Q_hat(valid_states),nlightbulb_problem(c).fit.Q_star(valid_states,1))
set(gca,'FontSize',16)
xlabel(['$\hat{Q}=',modelEquation(feature_names(selected_features),roundsd(w,4)),'$'],...
    'Interpreter','LaTeX','FontSize',16)
ylabel('$Q^\star$','FontSize',16,'Interpreter','LaTeX')
title(['Bayesian SARSA learns Q-function of n-lightbulb meta-MDP, R^2=',num2str(roundsd(R2(c),4))],'FontSize',16)
saveas(fig_Q,['../../results/figures/QFitnProblemBayesianSARSA_c',int2str(c),'.fig'])
saveas(fig_Q,['../../results/figures/QFitnProblemBayesianSARSA_c',int2str(c),'.png'])
