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
    
    for i=1:nr_states
        st = S(i,:);
        
        t = sum(st);
        mvoc = 1/(t*(t+1))*(st(1)*(max(st(1)+1,st(2))-min(st(1)+1,st(2))) + st(2)*(max(st(1),st(2)+1)-min(st(1),st(2)+1)));
        voc1(i) = mvoc-max(st)/sum(st)-cost+min(st)/sum(st);
        
        vpi(i) = valueOfPerfectInformationBernoulli(st(1),st(2),rewardCorrect,rewardIncorrect);
        
        voc30(i) = Q_star(i) - max(st)/sum(st)+min(st)/sum(st);
    end

    X = cat(2,vpi,voc1,bias); feature_names={'VPI','VOC_1','1'};
    [w,wint,r,rint,stats] = regress(voc30,X);
    voc_hat=X*w;
    
    E_guess=max(S,[],2)./sum(S,2) - min(S,[],2)./sum(S,2);
    
    Q_hat(:,1)=voc_hat+E_guess;
    Q_hat(:,2)=E_guess;
    Q_hat(end,:) = zeros(2,1);
    V_hat=max(Q_hat,[],2);

    valid_states=[1:nr_states]';
    
    Q_star=getQFromV(lightbulb_mdp(c).v_star,lightbulb_mdp(c).T,lightbulb_mdp(c).R);
    R2=corr(Q_star(valid_states,1),Q_hat(valid_states))^2;
    
    load ../results/lightbulb_problem
    lightbulb_problem(c).mdp=lightbulb_mdp(c);
    lightbulb_problem(c).fit.w=w;
    lightbulb_problem(c).fit.Q_star=Q_star;
    lightbulb_problem(c).fit.Q_hat=Q_hat;
    lightbulb_problem(c).fit.R2=R2;
    lightbulb_problem(c).fit.feature_names=feature_names;
    lightbulb_problem(c).fit.features=X;
end
save('../results/lightbulb_problem.mat','lightbulb_problem')