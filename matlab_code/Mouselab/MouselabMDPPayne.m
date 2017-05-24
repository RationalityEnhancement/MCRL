classdef MouselabMDPPayne < MDP
    %Model of Experiment 1 by Payne et al. (1988)
    
    properties
        min_payoff=0.01; %in $
        max_payoff=9.99; %in $
        mean_payoff=5;
        std_payoff=9.98/sqrt(12);
        nr_outcomes;
        nr_gambles;
        nr_cells;
        nr_possible_payoffs;
        payoff_values;
        p_payoff_values;
        p_nr_gambles=[0,0,0,1,0];
        p_nr_outcomes=[0,0,0,1,0];
        payoff_matrix;
        clicks_per_cell;
        total_nr_clicks;
        remaining_nr_clicks;
        outcome_probabilities;
        non_compensatoriness;
        alpha_clicks_per_cell=1;
        beta_clicks_per_cell=0;
        add_pseudorewards;
        pseudoreward_type;
        time_budgets=[inf,15]; %time budget per choice in seconds
        time_per_click=1.22; %seconds per acquisition
        time_cost_per_sec=7/3600*40; %8$/h in points/sec
        p_high_dispersion=0.50;
        time_pressure;
        p_time_pressure=0.50;
        high_dispersion;
        alpha_high_dispersion=0.50;
        action_features;
        payoff_ranges;
    end
    
    methods
        function mdp=MouselabMDPPayne(add_pseudorewards,pseudoreward_type,payoff_ranges)
            
            mdp.gamma=1;

            if not(exist('add_pseudorewards','var'))
                add_pseudorewards=false;
            end
            mdp.add_pseudorewards=add_pseudorewards;
            
            if not(exist('pseudoreward_type','var'))
                pseudoreward_type='';
            end
            mdp.pseudoreward_type=pseudoreward_type;
            
            if not(exist('payoff_ranges','var'))
                payoff_ranges=[0.01, 9.99];
            end
        end
        
        function [state,mdp]=newEpisode(mdp)
            mdp.nr_gambles=sampleDiscreteDistributions(mdp.p_nr_gambles,1);
            mdp.nr_outcomes=sampleDiscreteDistributions(mdp.p_nr_outcomes,1);

            max_nr_clicks=mdp.nr_outcomes*mdp.nr_gambles+1;
            mdp.time_budgets(1)=(max_nr_clicks+1)*mdp.time_per_click;
            
            %mdp.outcome_probabilities=mdp.sampleOutcomeProbabilities();
            
            %set payoff range
            nr_payoff_ranges=size(mdp.payoff_ranges,1);
            range_index=randi(nr_payoff_ranges);
            mdp=mdp.setPayoffRange(mdp.payoff_ranges(range_index,1),...
                mdp.payoff_ranges(range_index,2));
            
            mdp.payoff_values=mdp.min_payoff:0.01:mdp.max_payoff;
            mdp.nr_possible_payoffs=numel(mdp.payoff_values);
            mdp.p_payoff_values=discreteNormalPMF(mdp.payoff_values,...
                mdp.mean_payoff,mdp.std_payoff);
            mdp.time_pressure=rand()<mdp.p_time_pressure;
            
            %mdp.payoff_matrix=mdp.samplePayoffMatrix();
            %mdp.outcome_probabilities=mdp.sampleOutcomeProbabilities();
            
            
            mdp.clicks_per_cell=betarnd(mdp.alpha_clicks_per_cell,mdp.beta_clicks_per_cell);
            mdp.nr_cells=mdp.nr_gambles*mdp.nr_outcomes;
                                    
            [state,mdp]=mdp.sampleS0();
            mdp=mdp.setActions(state);
            mdp.nr_actions=numel(mdp.actions);
            
            [~,mdp]=mdp.extractStateActionFeatures(state,mdp.actions(1));
        end
        
        function [state,mdp]=sampleS0(mdp)
            
            %sample number of outcomes and number of gambles
            state.observations=NaN(mdp.nr_outcomes,mdp.nr_gambles);
            
            state.decision=NaN;
            state.outcome=NaN;
            
            if mdp.time_pressure
                state.time_left=mdp.time_budgets(2);
            else
                state.time_left=mdp.time_budgets(1);
            end
            
            mdp.payoff_matrix=mdp.samplePayoffMatrix();
            
            mdp.high_dispersion=rand()<mdp.p_high_dispersion;
                        
            mdp.outcome_probabilities=mdp.sampleOutcomeProbabilities();
            
            state.mu=mdp.mean_payoff*ones(mdp.nr_gambles,1);
            state.sigma=mdp.std_payoff*sqrt(sum(mdp.outcome_probabilities.^2))*ones(mdp.nr_gambles,1);
            
            mdp=mdp.setActions(state);
            
        end
        
        function true_or_false=isTerminalState(mdp,state)
            true_or_false=not(isnan(state.decision));
        end
        
        function [ER,PR]=expectedReward(mdp,state,action)
            
            time_cost=mdp.time_cost_per_sec*mdp.time_per_click;
            
            if action.is_decision
                ER=state.mu(action.gamble)-time_cost;
            else
                ER=-time_cost;
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
            
            time_cost=mdp.time_cost_per_sec*mdp.time_per_click;
            
            if not(action.is_decision)
                if isnan(state.observations(action.outcome,action.gamble)) %not observed yet
                    next_state=mdp.addObservation(state,action.outcome,action.gamble,...
                        mdp.payoff_matrix(action.outcome,action.gamble));
                else %the box had already been opened before -- why subtract a click for this if observations remain revealed?
                    next_state=state;
                    next_state.time_left=state.time_left-mdp.time_per_click;
                end
                
                r=-time_cost;
                next_state.outcome=NaN;
            else
                %a decision has been made
                next_state=state;
                next_state.decision=action.gamble;
                o=sampleDiscreteDistributions(mdp.outcome_probabilities',1);
                r=mdp.payoff_matrix(o,action.gamble)-time_cost;
                
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
                %[next_state,mdp]=mdp.newEpisode();
                                
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
                    next_states(o).time_left=state.time_left-mdp.time_per_click;
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
                    next_states(1).time_left=state.time_left-mdp.time_per_click;
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
                    payoffs(o,g)=max(mdp.min_payoff,min(mdp.max_payoff,round(v,2)));
                end
            end
            
        end
        
        function outcome_probabilities=sampleOutcomeProbabilities(mdp)
            %sample outcome probabilities
            if mdp.high_dispersion
                outcome_probabilities=stickBreaking(mdp.nr_outcomes,mdp.alpha_high_dispersion);
                while or(max(outcome_probabilities)<0.5,max(outcome_probabilities)>0.99)
                    outcome_probabilities=stickBreaking(mdp.nr_outcomes,mdp.alpha_high_dispersion);
                end
            else
                temp=rand(mdp.nr_outcomes,1);
                outcome_probabilities=temp/sum(temp);
            end
        end
        
        function [actions,mdp]=getActions(mdp,state)
            mdp=mdp.setActions(state);
            actions=mdp.actions;
        end
        
        function [mdp,actions]=setActions(mdp,state)
            
            nr_decisions=1;%mdp.nr_gambles;
            nr_acquisitions=mdp.nr_gambles*mdp.nr_outcomes;
            nr_actions=nr_decisions+nr_acquisitions-sum(not(isnan(state.observations(:))));
            actions=repmat(struct('is_decision',true,'outcome',1,'gamble',1),[nr_actions,1]);
            
            %When the agent decides, it always chooses the gamble with the
            %highest estimate EV.
            a=1;
            actions(a).is_decision=true;
            actions(a).outcome=NaN;
            actions(a).gamble=argmax(state.mu);
            %{
            %decisions            
            for a=1:nr_decisions
                mdp.actions(a).is_decision=true;
                mdp.actions(a).outcome=NaN;
                mdp.actions(a).gamble=a;
            end
            %}
            
            %acquisitions
            %acquisitions are possible only when the agent has more than 1
            %click left.            
            if state.time_left>mdp.time_per_click
                a=nr_decisions;
                for o=1:mdp.nr_outcomes
                    for g=1:mdp.nr_gambles
                        if isnan(state.observations(o,g)) %If this cell has not been observed yet, then allow it to be observed.
                            a=a+1;
                            actions(a).is_decision=false;
                            actions(a).outcome=o;
                            actions(a).gamble=g;
                        end
                    end
                end
            end
            
            mdp.actions=actions;
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
            
            
            %update the remaining nr of clicks
            next_state.time_left=state.time_left-mdp.time_per_click;
            
            %update belief state
            unobserved=isnan(state.observations);
            
            if unobserved(outcome,gamble)
                
                %update observations
                next_state.observations(outcome,gamble)=payoff;
                
                next_state.mu(gamble)=state.mu(gamble)+...
                    mdp.outcome_probabilities(outcome)*payoff-...
                    mdp.outcome_probabilities(outcome)*mdp.mean_payoff;
                
                unobserved_outcomes=isnan(next_state.observations(:,gamble));
                next_state.sigma(gamble)=mdp.std_payoff*...
                    sqrt(sum(mdp.outcome_probabilities(unobserved_outcomes).^2));
            end
            
        end
        
        function [state_action_features,mdp]=extractStateActionFeatures(mdp,state,action)
            state_features=mdp.extractStateFeatures(state);
            action_features=mdp.extractActionFeatures(state,action);
            
            state_action_features=[1;state_features;action_features];
            
            is_action_feature=[zeros(1+numel(state_features),1); ones(numel(action_features),1)];
            mdp.action_features=find(is_action_feature);
            
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
                nr_observations=sum(not(isnan(state.observations(:))));
                state_features(:,i)=[nr_observations;
                    expected_optimum; state.mu(a_star);state.sigma(a_star);...
                    sigma_maximum; ...
                    state.mu(b);state.sigma(b);...
                    %max(state.sigma(:));
                    expected_regret; state.time_left%; state.time_left
                    %;size(state.observations,2)
                    ];
                %; max(myopic_VOC(:))
            end
        end
        
        function action_features=extractActionFeatures(mdp,state,action)
                        
            if action.is_decision
                %ER=state.mu(action.gamble);
                VOC=0;
                sigma=0;
                probability=0;
                delta_mu=0;
                last_click_decision=state.time_left<mdp.time_per_click;
                %early_decision=double(not(last_click_decision));
                %indecision=0;
                observation=0;
            else
                %ER=0;
                VOC=mdp.myopicVOC(state,action);
                sigma=state.sigma(action.gamble);
                probability=mdp.outcome_probabilities(action.outcome);
                delta_mu=max(state.mu)-state.mu(action.gamble);                
                %indecision=state.time_left<=mdp.total_nr_clicks;
                observation=1;
                %early_decision=0;
                last_click_decision=0;
            end
            
            action_features=[%ER
                VOC;sigma;probability; delta_mu; last_click_decision; ...
                %early_decision;                
                observation
                ];
            
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
            
            if and(state.time_left>mdp.time_per_click,not(action.is_decision))
                
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
            elseif and(state.time_left<mdp.time_per_click, not(action.is_decision)) 
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
        
        function mdp=setPayoffRange(mdp,min_payoff,max_payoff)
            mdp.max_payoff=max_payoff;
            mdp.min_payoff=min_payoff;
            mdp.mean_payoff=(mdp.max_payoff+mdp.min_payoff)/2;
            mdp.payoff_values=mdp.min_payoff:0.01:mdp.max_payoff;
            payoff_range=mdp.max_payoff-mdp.min_payoff;
            mdp.std_payoff=payoff_range/2/1.96;
            mdp.p_payoff_values=discreteNormalPMF(mdp.payoff_values,...
                mdp.mean_payoff,mdp.std_payoff);
        end
        
    end
    
end