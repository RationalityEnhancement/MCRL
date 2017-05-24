addpath('./MatlabTools/')

costs=logspace(-3,-1,10);

load ../results/lightbulb_problem
nr_states=size(lightbulb_mdp(1).states,1);
S=lightbulb_mdp(1).states;


for c=1:numel(costs)
    cost = costs(c);
    voc1 = zeros(nr_states,1);
    vpi = zeros(nr_states,1);
    bias = ones(nr_states,1);
    voc30 = zeros(nr_states,1);
    
    Q_star=lightbulb_mdp(c).Q_star;
    
%     voc2 = zeros(nr_states,1);
%     stde=@(st) st(1)*st(2)/((st(1)+st(2))^2+(st(1)+st(2)+1));
%     stds = zeros(nr_states,1);
%     to = zeros(nr_states,1);
%     b = zeros(nr_states,1);
    
    for i=1:nr_states
        st = S(i,:);
        
%         stds(i) = stde(st);
%         to(i) = sum(st);
%         b(i) = max(st)/sum(st);
        
        t = sum(st);
        mvoc = 1/(t*(t+1))*(st(1)*max(st(1)+1,st(2)) + st(2)*max(st(1),st(2)+1));
        voc1(i) = mvoc-max(st)/sum(st)-cost;
        
%         voc2(i) = nvoc(3,st,cost);
        
%         vpi(i) = 1 - max(st)/sum(st);
        vpi(i) = valueOfPerfectInformationBernoulli(st(1),st(2));
        
        voc30(i) = Q_star(i) - max(st)/sum(st);
    end

    X = cat(2,vpi,voc1,bias); feature_names={'VPI','VOC_1','1'};
    [w,wint,r,rint,stats] = regress(voc30,X);
    voc_hat=X*w;
    figure();
    scatter(voc_hat,voc30);
    title(num2str(stats(1)));
    
    sign_disagreement=find(sign(voc_hat).*sign(voc30)==-1)
    numel(sign_disagreement)/numel(voc30)    
    max(voc30(sign_disagreement))
    
    E_guess=max(S,[],2)./sum(S,2);
    
    
    %% Plot fit to Q-function
    
    Q_hat(:,1)=voc_hat+E_guess;
    Q_hat(:,2)=E_guess;
    Q_hat(end,:) = zeros(2,1);
    V_hat=max(Q_hat,[],2);
    
%     valid_states=and(sum(S,2)<=30,sum(S,2)>0);
    valid_states=[1:nr_states]';
    
    Q_star=getQFromV(lightbulb_mdp(c).v_star,lightbulb_mdp(c).T,lightbulb_mdp(c).R);
    R2=corr(Q_star(valid_states,1),Q_hat(valid_states))^2;
    
    fig_Q=figure();
    scatter(Q_hat(valid_states),Q_star(valid_states,1))
    set(gca,'FontSize',16)
    xlabel(modelEquation(feature_names,w),'FontSize',16)
    ylabel('$Q^\star$','FontSize',16,'Interpreter','LaTeX')
    title(['Linear Fit to Q-function of 1-lightbulb meta-MDP, R^2=',num2str(R2)],'FontSize',16)
    saveas(fig_Q,'../results/figures/QFitToyProblem.fig')
    saveas(fig_Q,'../results/figures/QFitToyProblem.png')
    
    load ../results/lightbulb_problem
    lightbulb_problem(c).mdp=lightbulb_mdp(c);
    lightbulb_problem(c).fit.w=w;
    lightbulb_problem(c).fit.Q_star=Q_star;
    lightbulb_problem(c).fit.Q_hat=Q_hat;
    lightbulb_problem(c).fit.R2=R2;
    lightbulb_problem(c).fit.feature_names=feature_names;
    lightbulb_problem(c).fit.features=X;
    lightbulb_problem(c).optimal_PR=lightbulb_problem(c).mdp.optimal_PR;
end
save('../results/lightbulb_fit.mat','lightbulb_problem')