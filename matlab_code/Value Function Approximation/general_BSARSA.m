%evaluate and tune Bayesian-SARSAQ with the features identified by linear
%regression
%change to folder that contains this file (Value Function Approximation)

addpath('../MatlabTools/') %change to your directory for MatlabTools
addpath('../generalMDP/')
addpath('../Supervised/')

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

nr_episodes = 1000;
rep = 1;

feature_names={'VPI','VOC_1','E[R|guess,b]','1'};
selected_features=[1;2;3];

nr_features=numel(selected_features);

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

            w=glm.mu_n;        

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

            nlightbulb_problem.bsarsa.w_BSARSA=w;
            nlightbulb_problem.bsarsa.Q_hat_BSARSA=Q_hat;
            nlightbulb_problem.bsarsa.V_hat_BSARSA=V_hat;
            nlightbulb_problem.bsarsa.R2_BSARSA=R2;

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
        end
    end
end
nlightbulb_problem.approximate_PRs=approximate_PR;
save(['../../results/', num2str(nr_arms),'lightbulb_fit.mat'],'nlightbulb_problem','-v7.3')