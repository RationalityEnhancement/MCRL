addpath('./MatlabTools/')
addpath('../')

nr_arms= 2;
nr_balls = 6;
cost = 0.01;

exact

load(['../results/', num2str(nr_arms),'lightbulb_problem.mat'])

states = nlightbulb_mdp.states;
nr_states=size(states,1)-1;
S=states(1:nr_states,:);
nr_arms = size(states(1,:),2)/2;

nr_observations=sum(states,2)-2*nr_arms;
max_nr_observations=max(nr_observations); 

% valid_states=find(and(nr_observations<max_nr_observations,...
%     nr_observations>=0));
valid_states = [1:nr_states]';
s_vstates = size(valid_states);
n_vstates = s_vstates(1);


%% Fill in the regressors

Q_star=getQFromV(nlightbulb_mdp.v_star,nlightbulb_mdp.T,nlightbulb_mdp.R);
voc1 = zeros(nr_states,nr_arms);
vpi = zeros(nr_states,nr_arms);
voc = zeros(nr_states,nr_arms);
ers = zeros(nr_states,nr_arms);
bias = ones(nr_states*nr_arms,1);
state_action = zeros(nr_states,nr_arms);
count = 0;

for k=1:numel(valid_states)
    i = valid_states(k);
    st = S(i,:);  
    st_m = reshape(st,2,nr_arms)';
    er = max( st_m(:,1) ./ sum(st_m,2));
    for j=1:nr_arms
        count = count +1;
        state_action(i,j) = count;
        ers(i,j) = er;
        vpi(i,j) = valueOfPerfectInformationMultiArmBernoulli(st_m(:,1),st_m(:,2),j);
        voc1(i,j) = VOC1MultiArmBernoulli(st_m(:,1),st_m(:,2),j,cost);
        voc(i,j) = Q_star(i,j) - cost - er;
    end
end
    
%% Regression

vpiv = vpi(valid_states,:)';
voc1v = voc1(valid_states,:)';
ersv = ers(valid_states,:)';
bias_size = nr_arms*n_vstates;
X = cat(2,voc1v(:),vpiv(:),bias(1:bias_size));
feature_names={'VOC1','VPI','1'};

voc_valid_states = voc(valid_states,:)';
voc_valid_states = voc_valid_states(:);

[w,wint,r,rint,stats] = regress(voc_valid_states,X);
voc_hat=X*w;
figure();
scatter(voc_hat,voc_valid_states);
title(['R^2=',num2str(stats(1))]);
xlabel('Predicted VOC','FontSize',16)
ylabel('VOC','FontSize',16)

sign_disagreement = find(sign(voc_hat).*sign(voc_valid_states)==-1);
sign_disagreement_rate = numel(sign_disagreement)/numel(voc_valid_states);
max_sign_disagreement = max(voc_valid_states(sign_disagreement));

%% Plot fit to Q-function

Q_hat=reshape(voc_hat,nr_arms,n_vstates)' + ersv' + cost;
Q_hat=[Q_hat,ersv(1,:)'];
V_hat=max(Q_hat,[],2);

qh = Q_hat(valid_states,:);
qs = Q_star(valid_states,:);
R2=corr(qs(:),qh(:))^2;

fig_Q=figure();
scatter(qh(:),qs(:))
set(gca,'FontSize',16)
xlabel(modelEquation(feature_names,w),'FontSize',16)
ylabel('$Q^\star$','FontSize',16,'Interpreter','LaTeX')
title(['Linear Fit to Q-function of n-lightbulb meta-MDP, R^2=',num2str(R2)],'FontSize',16)
saveas(fig_Q,'../results/figures/QFitNBulbs.fig')
saveas(fig_Q,'../results/figures/QFitNBulbs.png')

nlightbulb_problem.mdp=nlightbulb_mdp;
nlightbulb_problem.fit.w=w;
nlightbulb_problem.fit.Q_star=qs;
nlightbulb_problem.fit.Q_hat=Q_hat;
nlightbulb_problem.fit.R2=R2;
nlightbulb_problem.fit.feature_names=feature_names;
nlightbulb_problem.fit.features=X;
%     nlightbulb_problem(c).optimal_PR=nlightbulb_problem(c).mdp.optimal_PR;

disp(['../results/', num2str(nr_arms),'lightbulb_fit.mat'])
save(['../results/', num2str(nr_arms),'lightbulb_fit.mat'],'nlightbulb_problem','-v7.3')