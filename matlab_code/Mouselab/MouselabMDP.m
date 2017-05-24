classdef MouselabMDP < MDP
    
    properties
        min_payoff=-100;
        max_payoff=100;
        mean_payoff=0;
        std_payoff=40;
        nr_outcomes;
        nr_gambles;
        nr_cells;
        nr_possible_payoffs;
        payoff_values;
        p_payoff_values;
        p_nr_gambles=[0,1/4,1/4,1/4,1/4];
        p_nr_outcomes=[0,1/4,1/4,1/4,1/4];
        payoff_matrix;
        clicks_per_cell;
        total_nr_clicks;
        remaining_nr_clicks;
        outcome_probabilities;
        non_compensatoriness;
        alpha_clicks_per_cell;
        beta_clicks_per_cell;
        add_pseudorewards;
        pseudoreward_type;
    end
    
    methods
        function mdp=MouselabMDP(non_compensatoriness,alpha_clicks_per_cell,beta_clicks_per_cell,add_pseudorewards,pseudoreward_type)
            
            mdp.gamma=1;
            mdp.non_compensatoriness=non_compensatoriness;
            mdp.alpha_clicks_per_cell=alpha_clicks_per_cell;
            mdp.beta_clicks_per_cell=beta_clicks_per_cell;
            
            if not(exist('add_pseudorewards','var'))
                add_pseudorewards=false;
            end
            mdp.add_pseudorewards=add_pseudorewards;
            
            if not(exist('pseudoreward_type','var'))
                pseudoreward_type='';
            end
            mdp.pseudoreward_type=pseudoreward_type;
        end
        
        function [state,mdp]=newEpisode(mdp)
            mdp.nr_gambles=sampleDiscreteDistributions(mdp.p_nr_gambles,1);
            mdp.nr_outcomes=sampleDiscreteDistributions(mdp.p_nr_outcomes,1);
            
            %mdp.outcome_probabilities=mdp.sampleOutcomeProbabilities();
            
            mdp.payoff_values=mdp.min_payoff:mdp.max_payoff;
            mdp.nr_possible_payoffs=numel(mdp.payoff_values);
            mdp.p_payoff_values=discreteNormalPMF(mdp.payoff_values,...
                mdp.mean_payoff,mdp.std_payoff);
            
            %mdp.payoff_matrix=mdp.samplePayoffMatrix();
            %mdp.outcome_probabilities=mdp.sampleOutcomeProbabilities();
            
            
            mdp.clicks_per_cell=betarnd(mdp.alpha_clicks_per_cell,mdp.beta_clicks_per_cell);
            mdp.nr_cells=mdp.nr_gambles*mdp.nr_outcomes;
            mdp.total_nr_clicks=round(mdp.clicks_per_cell*(mdp.nr_cells))+1;
            mdp.remaining_nr_clicks=mdp.total_nr_clicks;
            
            [state,mdp]=mdp.sampleS0(mdp.total_nr_clicks);
            mdp=mdp.setActions(state);
            mdp.nr_actions=numel(mdp.actions);
            
            
        end
        
        function [state,mdp]=sampleS0(mdp,nr_clicks_left)
            
            %sample number of outcomes and number of gambles
            state.observations=NaN(mdp.nr_outcomes,mdp.nr_gambles);
            
            if exist('nr_clicks_left','var')
                state.remaining_nr_clicks=nr_clicks_left;
            else
                state.remaining_nr_clicks=mdp.total_nr_clicks;
            end
            state.decision=NaN;
            state.outcome=NaN;
            
            mdp.payoff_matrix=mdp.samplePayoffMatrix();
            mdp.outcome_probabilities=mdp.sampleOutcomeProbabilities();
            
            mdp=mdp.setActions(state);
            
            state.mu=zeros(mdp.nr_gambles,1);
            state.sigma=mdp.std_payoff*sqrt(sum(mdp.outcome_probabilities.^2))*ones(mdp.nr_gambles,1);
            
        end
        
        function true_or_false=isTerminalState(mdp,state)
            true_or_false=state.remaining_nr_clicks==0;
        end
        
        function [ER,PR]=expectedReward(mdp,state,action)
            if action.is_decision
                ER=state.mu(action.gamble);
            else
                ER=0;
            end
            
            if mdp.add_pseudorewards
                
                if strcmp(mdp.pseudoreward_type,'myopicVOC')
                    PR=mdp.myopicVOC(state,action);
                elseif strcmp(mdp.pseudoreward_type,'regretReduction')
                    PR=mdp.expectedRegretReduction(state,action);                   
                end
            else
                PR=0;
            end
        end
        
        function [r,next_state,PR]=simulateTransition(mdp,state,action)
            
            if not(action.is_decision)
                if isnan(state.observations(action.outcome,action.gamble)) %not observed yet
                    next_state=mdp.addObservation(state,action.outcome,action.gamble,...
                        mdp.payoff_matrix(action.outcome,action.gamble));
                else %the box had already been opened before -- why subtract a click for this if observations remain revealed?
                    next_state=state;
                    next_state.remaining_nr_clicks=state.remaining_nr_clicks-1;
                end
                
                r=0;
                next_state.outcome=NaN;
            else
                %a decision has been made
                next_state=state;
                next_state.decision=action.gamble;
                o=sampleDiscreteDistributions(mdp.outcome_probabilities',1);
                r=mdp.payoff_matrix(o,action.gamble);
                
                %{
                %information state after the choice but before the next
                episode
                next_state.decision=action.gamble;
                next_state.outcome=o;
                next_state.observations(o,action.gamble)=r;
                next_state.mu(action.gamble)=mdp.payoff_matrix(next_state.outcome,action.gamble);
                next_state.sigma(action.gamble)=0;
                
                %gambles for which the sampled outcome has been observed
                next_state.mu(not(isnan(next_state.observations(o,:))))=mdp.payoff_matrix(o,not(isnan(next_state.observations(o,:))));
                next_state.sigma(not(isnan(next_state.observations(o,:))))=zeros(sum(not(isnan(next_state.observations(o,:)))),1);
                
                %gambles for which the sampled outcome has not been
                %observed
                next_state.mu(isnan(next_state.observations(o,:)))=zeros(sum(isnan(next_state.observations(o,:))),1);
                next_state.sigma(isnan(next_state.observations(o,:)))=mdp.std_payoff;
                %}
                %information state at the beginning of the next episode
                [next_state,mdp]=mdp.newEpisode();
                                
                %end the episode
                next_state.remaining_nr_clicks=0;
            end
            
            if mdp.add_pseudorewards
                if strcmp(mdp.pseudoreward_type,'myopicVOC')
                    PR=mdp.myopicVOC(state,action);
                elseif strcmp(mdp.pseudoreward_type,'regretReduction')
                    PR=mdp.regretReduction(state,next_state);
                end
            else
                PR=0;
            end
                        
        end
        
        function [next_states,p_next_states]=predictNextState(mdp,state,action)
            
            if action.is_decision
                next_states=repmat(state,[mdp.nr_outcomes,1]);
                for o=1:mdp.nr_outcomes
                    next_states(o).decision=action.gamble;
                    next_states(o).outcome=o;
                    next_states(o).nr_clicks_left=state.remaining_nr_clicks-1;
                end
                
                p_next_states=mdp.outcome_probabilities;
            else
                
                if isnan(state.observations(action.outcome,action.gamble))
                    next_states=repmat(state,[mdp.nr_possible_payoffs,1]);
                    %inspect another payoff
                    for v=1:mdp.nr_possible_payoffs
                        next_states(v)=mdp.addObservation(state,action.outcome,...
                            action.gamble,mdp.payoff_values(v));
                    end
                    p_next_states=mdp.p_payoff_values;
                else
                    next_states=repmat(state,[1,1]);
                    next_states(1).remaining_nr_clicks=state.remaining_nr_clicks-1;
                    p_next_states=1;
                end
                
            end
            
        end
        
        function payoffs=samplePayoffMatrix(mdp)
            %sample values for each cell of the payoff matrix
            payoffs=NaN(mdp.nr_outcomes,mdp.nr_gambles);
            for o=1:mdp.nr_outcomes
                for g=1:mdp.nr_gambles
                    v=mdp.mean_payoff+mdp.std_payoff*randn();
                    payoffs(o,g)=max(mdp.min_payoff,min(mdp.max_payoff,round(v)));
                end
            end
            
        end
        
        function outcome_probabilities=sampleOutcomeProbabilities(mdp)
            %sample outcome probabilities
            outcome_probabilities=stickBreaking(mdp.nr_outcomes,mdp.non_compensatoriness);
        end
        
        function [actions,mdp]=getActions(mdp,state)
            mdp=mdp.setActions(state);
            actions=mdp.actions;
        end
        
        function [mdp,actions]=setActions(mdp,state)
            
            nr_decisions=mdp.nr_gambles;
            nr_acquisitions=mdp.nr_gambles*mdp.nr_outcomes;
            nr_actions=nr_decisions+nr_acquisitions;
            mdp.actions=repmat(struct('is_decision',true,'outcome',1,'gamble',1),[nr_actions,1]);
            
            %decisions
            for a=1:nr_decisions
                mdp.actions(a).is_decision=true;
                mdp.actions(a).outcome=NaN;
                mdp.actions(a).gamble=a;
            end
            
            %acquisitions
            %acquisitions are possible only when the agent has more than 1
            %click left.
            if state.remaining_nr_clicks>1
                a=nr_decisions;
                for o=1:mdp.nr_outcomes
                    for g=1:mdp.nr_gambles
                        if isnan(state.observations(o,g)) %If this cell has not been observed yet, then allow it to be observed.
                            a=a+1;
                            mdp.actions(a).is_decision=false;
                            mdp.actions(a).outcome=o;
                            mdp.actions(a).gamble=g;
                        end
                    end
                end
            end
            
            mdp.nr_actions=numel(mdp.actions);
            actions=mdp.actions;
            
        end                
        
        function PR=getPseudoRewards(mdp,state)
            
            a=0;
            for o=1:mdp.nr_outcomes
                for g=1:mdp.nr_gambles
                    a=a+1;
                    action.is_decision=false;
                    action.outcome=o;
                    action.gamble=g;
                    
                    PR(o,g)=mdp.myopicVOC(state,action);
                    
                end
            end
            
        end
        
        function next_state=addObservation(mdp,state,outcome,gamble,payoff)
            
            next_state=state;
            
            %update observations
            next_state.observations(outcome,gamble)=payoff;
            
            %update the remaining nr of clicks
            next_state.remaining_nr_clicks=state.remaining_nr_clicks-1;
            
            %update belief state
            unobserved=isnan(state.observations);
            
            if unobserved(outcome,gamble)
                next_state.mu(gamble)=state.mu(gamble)+...
                    mdp.outcome_probabilities(outcome)*payoff;
                
                unobserved_outcomes=isnan(next_state.observations(:,gamble));
                next_state.sigma(gamble)=mdp.std_payoff*...
                    sqrt(sum(mdp.outcome_probabilities(unobserved_outcomes).^2));
            end
            
        end
        
        function state_action_features=extractStateActionFeatures(mdp,state,action)
            state_features=mdp.extractStateFeatures(state);
            action_features=mdp.extractActionFeatures(state,action);
            
            state_action_features=[state_features;action_features];
            
            if any(isnan(state_action_features))
                throw(MException('MException:isNaN','features are NaN'))
            end
        end
        
        function state_features=extractStateFeatures(mdp,states)
            
            for i=1:numel(states)
                state=states(i);
                a_star=argmax(state.mu);
                sigma_maximum=sqrt(sum(state.sigma.^2)); %upper bound on the standard deviation of the maximum
                [~,b]=kthLargestElement(state.mu,2);
                
                %{
                myopic_VOC=zeros(mdp.nr_outcomes,mdp.nr_gambles);
                for g=1:mdp.nr_gambles
                    for o=1:mdp.nr_outcomes
                        action.gamble=g; action.outcome=o;
                        action.is_decision=false;
                        myopic_VOC(o,g)=mdp.myopicVOC(state,action);
                    end
                end
                %}

                [expected_regret,expected_optimum]=mdp.expectedRegret(state);                                
                
                state_features(:,i)=[state.mu(a_star);state.sigma(a_star);...
                    sigma_maximum; state.remaining_nr_clicks;...
                    state.mu(b);state.sigma(b);...
                    %max(state.sigma(:));
                    expected_regret;
                    expected_optimum
                    %;size(state.observations,2)
                    ];
                %; max(myopic_VOC(:))
            end
        end
        
        function action_features=extractActionFeatures(mdp,state,action)
                        
            if action.is_decision
                ER=state.mu(action.gamble);
                VOC=0;
                sigma=0;
                probability=0;
                delta_mu=0;
                last_click_decision=state.remaining_nr_clicks==1;
                early_decision=state.remaining_nr_clicks>1;
                indecision=0;
            else
                ER=0;
                VOC=mdp.myopicVOC(state,action);
                sigma=state.sigma(action.gamble);
                probability=mdp.outcome_probabilities(action.outcome)*mdp.std_payoff;
                delta_mu=state.mu(action.gamble)-max(state.mu);
                indecision=state.remaining_nr_clicks==1;
                early_decision=0;
                last_click_decision=0;
            end
            
            action_features=[ER;VOC;sigma;probability;delta_mu;...
                early_decision;last_click_decision];
            
        end
        
        function [expected_regret,expected_optimum]=expectedRegret(mdp,state)
            a_star=argmax(state.mu); %if >1 mu's are max, shouldn't a_star be selected randomly (not the first)
            x=mvnrnd(state.mu,diag(state.sigma.^2),100);  % this should come from a truncated normal
            regret=max(x,[],2)-x(:,a_star);
            expected_regret=mean(regret);
            expected_optimum=mean(max(x,[],2));
%             disp(['mu: ',num2str(state.mu(1)),' sigma: ',num2str(state.sigma(1)),' len: ',num2str(length(state.mu))])
        end
        
        function VOC=myopicVOC(mdp,state,action)
            
            if and(state.remaining_nr_clicks>1,not(action.is_decision))
                
                if isnan(state.observations(action.outcome,action.gamble))
                    max_prior_EV=max(state.mu);
                    
                    [next_states,p_next_states]=mdp.predictNextState(state,action);
                    
                    a_old=argmax(state.mu);
                    delta_EV=zeros(numel(next_states),1);
                    for s=1:numel(next_states)
                        max_posterior_EV=max(next_states(s).mu);
                        delta_EV(s)=max_posterior_EV-next_states(s).mu(a_old);
                    end
                    
                    VOC=dot(p_next_states,delta_EV);
                else
                    VOC=0;
                end
            elseif and(state.remaining_nr_clicks==1, not(action.is_decision)) 
                VOC=-max(state.mu);
            elseif action.is_decision
                VOC=0;
            end
        end
        
        function regret_reduction=regretReduction(mdp,s_old,s_new)
            %pseudo-reward derived from the potential function
            %phi(s)=E[max EV]-expected_regret
            [expected_regret_old_state,expected_maximum_old]=mdp.expectedRegret(s_old);
            [expected_regret_new_state,expected_maximum_new]=mdp.expectedRegret(s_new);
            %Phi_new=expected_maximum_new-expected_regret_new_state;
            %Phi_old=expected_maximum_old-expected_regret_old_state;

            Phi_new=-expected_regret_new_state;
            Phi_old=-expected_regret_old_state;
            regret_reduction=Phi_new-Phi_old;
        end
        
    end
    
end