%evaluate and tune Bayesian-SARSAQ with the features identified by linear
%regression
%change to folder that contains this file (Value Function Approximation)
clear,close all,clc

addpath('../MatlabTools/') %change to your directory for MatlabTools
addpath('../generalMDP/')
addpath('../Supervised/')

nr_arms= 2;
nr_balls = 6;
cost = 0.01;

load(['../../results/', num2str(nr_arms),'lightbulb_fit.mat'])

states = nlightbulb_problem.mdp.states;
Q_star = nlightbulb_problem.mdp.Q_star;
nr_states=size(states,1);
disp(nr_states);
%% Hyperparameter Tuning

% mus=[[1;1;1],[0;1;1],[1;0;1],[1;1;0],[0;0;1],[0;1;0],[1;0;0],[0;0;0],[0.5;0.5;0.5]];
% sigmas = 0.1:0.1:0.3;
% % sigmas = 0.05:0.05:0.3;

mus = [1;0;1];
sigmas = 0.3;
gamma=1;

nr_episodes = 1500;
rep = 1;

feature_names={'VPI','VOC_1','E[R|guess,b]','1'};
selected_features=[1;2;3];

nr_features=numel(selected_features);

matws = zeros(size(mus,2),numel(sigmas),rep,3);
matr = zeros(size(mus,2),numel(sigmas),rep);
matm = zeros(size(mus,2),numel(sigmas),rep);

for m=1:size(mus,2)
    for sig=1:numel(sigmas)
        disp(num2str(mus(:,m)))
        disp(num2str(sigmas(sig)));
        for z=1:rep
            mdp=generalMDP(nr_arms,gamma,nr_features,cost,nr_balls);

            fexr=@(s,a,mdp) feature_extractor(s,a,mdp,selected_features);

            mdp.action_features=1:nr_features;

            sigma0=sigmas(sig);
            glm=BayesianGLM(nr_features,sigma0);
            glm.mu_0=mus(:,m);
            glm.mu_n=mus(:,m);
            [glm,avg_MSE,R_total]=BayesianSARSAQ(mdp,fexr,nr_episodes,glm);

            figure(),
            subplot(2,1,1)
            plot(smooth(avg_MSE,100))
            xlabel('Episode','FontSize',16)
            ylabel('Average MSE','FontSize',16)

            subplot(2,1,2)
            plot(smooth(R_total,100))
            xlabel('Episode','FontSize',16)
            ylabel('R_{total}','FontSize',16)

            w=glm.mu_n;
            figure()
            bar(w)
            ylabel('Learned Weights','FontSize',16)
            set(gca,'XTickLabel',feature_names(selected_features),'FontSize',16)

            %plot the corresponding fit to the Q-function

            for s=1:nr_states
                st = states(s,:);
                st_m = reshape(st,2,nr_arms)';
                er = max( st_m(:,1) ./ sum(st_m,2));
                for a=1:nr_arms+1          
                    F=fexr(st_m,a,mdp);
                    Q_hat(s,a)=F'*w;
                end
            end
            Q_hat(nr_states,:)=zeros(nr_arms+1,1);

            nr_arms = size(states(1,:),2)/2;

            nr_observations=sum(states,2)-2*nr_arms;
            max_nr_observations=max(nr_observations); 
 
%             valid_states=find(and(nr_observations<max_nr_observations,...
%                 nr_observations>=0));
            valid_states = [1:nr_states]';

            V_hat=max(Q_hat,[],2);
            qh = Q_hat(valid_states,:);
            qs = Q_star(valid_states,:);
            R2 = corr(qh(:),qs(:))^2;

            nlightbulb_problem.w_BSARSA=w;
            nlightbulb_problem.Q_hat_BSARSA=Q_hat;
            nlightbulb_problem.V_hat_BSARSA=V_hat;
            nlightbulb_problem.R2_BSARSA=R2;

            fig_Q=figure();
            scatter(Q_hat(valid_states),Q_star(valid_states,1))
            set(gca,'FontSize',16)
            xlabel(['$\hat{Q}=',modelEquation(feature_names(selected_features),roundsd(w,4)),'$'],...
                'Interpreter','LaTeX','FontSize',16)
            ylabel('$Q^\star$','FontSize',16,'Interpreter','LaTeX')
            title(['Bayesian SARSA learns Q-function of n-lightbulb meta-MDP, R^2=',num2str(roundsd(R2,4))],'FontSize',16)
            saveas(fig_Q,['../../results/figures/QFitnProblemBayesianSARSA_c',num2str(cost),'.fig'])
            saveas(fig_Q,['../../results/figures/QFitnProblemBayesianSARSA_c',num2str(cost),'.png'])

            %% Compute approximate PRs (without rewards)
            approximate_PR = nan(nr_states,nr_arms+1);
            for s=1:nr_states
                for a=1:nr_arms+1
                    next_s = find(not(nlightbulb_problem.mdp.T(s,:,a) == 0));
                    evp = 0;
                    for isp=1:numel(next_s)
                        sp = next_s(isp);
                        evp = evp + nlightbulb_problem.mdp.T(s,sp,a)*V_hat(sp);
                    end
                    approximate_PR(s,a) = evp-V_hat(s);
                end
            end

%             approximate_PR = nan(nr_states,nr_arms+1);
%             approximate_PR_Q = nan(nr_states,nr_arms+1);
%             for s=1:nr_states
%                 for a=1:nr_arms+1
%                     approximate_PR_Q(s,a) = Q_hat(s,a) - V_hat(s);
%                     next_s = find(not(nlightbulb_problem.mdp.T(s,:,a) == 0));
%                     evp = 0;
%                     for isp=1:numel(next_s)
%                         sp = next_s(isp);
%                         evp = evp + nlightbulb_problem.mdp.T(s,sp,a)*V_hat(sp);
%                     end
%                     approximate_PR(s,a) = evp-V_hat(s)+nlightbulb_problem.mdp.R(s,a);
%                 end
%             end
%             disp(num2str(max(abs(approximate_PR(:)-approximate_PR_Q(:)))));

            matws(m,sig,z,:) = w;
            matm(m,sig,z) = immse(qh(:),qs(:)); zeros(size(mus,2),numel(sigmas),rep);
            matr(m,sig,z) = corr(qh(:),qs(:))^2;
        end
    end
end
nlightbulb_problem.approximate_PRs=approximate_PR;
nlightbulb_problem.matws = matws;
nlightbulb_problem.matm = matm;
nlightbulb_problem.matr = matr;
save(['../../results/', num2str(nr_arms),'lightbulb_fit.mat'],'nlightbulb_problem','-v7.3')