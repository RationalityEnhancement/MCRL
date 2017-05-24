classdef MouselabMDPMetaMDPNIPS < MDP
    
    properties
        min_payoff;
        max_payoff;
        mean_payoff=4.5;
        std_payoff=10.6;
        nr_cells;
        nr_possible_payoffs;
        payoff_values;
        p_payoff_values;
        reward_function;
        rewards;
        cost_per_click; %each click costs 5 cents
        cost_per_operation=0.005; %each planning operation costs 1 cents
        total_nr_clicks;
        remaining_nr_clicks;
        add_pseudorewards;
        pseudoreward_type;
        object_level_MDPs;
        object_level_MDP;
        episode=0;
        action_features=1:3;%7:16;
        feature_names;
        next_states;
        p_next_states;
        query_state;
        query_computation;
        Q_blinkered;
        states_blinkered;
        sigma_values;
        mu_values;
        terminal_states=10:17;
        nonterminal_states=1:9;
        PR_feature_weights;        
    end
    
    methods
        function meta_MDP=MouselabMDPMetaMDPNIPS(add_pseudorewards,pseudoreward_type,mean_payoff,std_payoff,object_level_MDPs,cost_per_click)
            
            meta_MDP.mean_payoff=mean_payoff;
            meta_MDP.std_payoff=std_payoff;
            meta_MDP.min_payoff=meta_MDP.mean_payoff-3*meta_MDP.std_payoff;
            meta_MDP.max_payoff=meta_MDP.mean_payoff+3*meta_MDP.std_payoff;
            meta_MDP.nr_possible_payoffs=25;
            meta_MDP.payoff_values=linspace(meta_MDP.min_payoff,meta_MDP.max_payoff,...
                meta_MDP.nr_possible_payoffs);
            
            meta_MDP.payoff_values=linspace(meta_MDP.min_payoff,meta_MDP.max_payoff,meta_MDP.nr_possible_payoffs);
            meta_MDP.p_payoff_values=discreteNormalPMF(meta_MDP.payoff_values,...
                meta_MDP.mean_payoff,meta_MDP.std_payoff);            
            
            meta_MDP.object_level_MDPs=object_level_MDPs;
            meta_MDP.nr_cells=numel(object_level_MDPs(1).states);
            
            meta_MDP.gamma=1;
            
            if not(exist('add_pseudorewards','var'))
                add_pseudorewards=false;
            end
            meta_MDP.add_pseudorewards=add_pseudorewards;
            
            if not(exist('pseudoreward_type','var'))
                pseudoreward_type='';
            end
            meta_MDP.pseudoreward_type=pseudoreward_type;
            
            
            state_feature_names={'max mu','sigma(argmax mu(a))','E[max R]','STD[max R]',...
                    'mu(beta)', 'sigma(beta)'};
action_feature_names={'Expected regret','regret reduction','VOC',...
                'uncertainty reduction','sigma(R)','p(best action)','sigma(best action)',...
                'underplanning','complete planning','cost'};
            
            feature_names={'VPI','VOC','E[R|guess]'};%{state_feature_names{:}, action_feature_names{:}};
            
            meta_MDP.feature_names=feature_names;
            
            if exist('cost_per_click','var')
                meta_MDP.cost_per_click=cost_per_click;
            else
                meta_MDP.cost_per_click=0.05;
            end
            
            switch(meta_MDP.cost_per_click)
                case 0.01
                    meta_MDP.PR_feature_weights=struct('VPI',1.2065, 'VOC1', 2.1510, 'ER', 1.5298);
                case 1.60
                    meta_MDP.PR_feature_weights=struct('VPI',0.6118, 'VOC1', 1.2708, 'ER', 1.3215);
                case 2.80
                    meta_MDP.PR_feature_weights=struct('VPI',0.6779, 'VOC1', 0.7060, 'ER', 1.2655);
            end
        end
        
        function [state,meta_MDP]=newEpisode(meta_MDP)
            
            meta_MDP.episode=meta_MDP.episode+1;
            
            meta_MDP.payoff_values=linspace(meta_MDP.min_payoff,meta_MDP.max_payoff,meta_MDP.nr_possible_payoffs);
            
            meta_MDP.p_payoff_values=discreteNormalPMF(meta_MDP.payoff_values,...
                meta_MDP.mean_payoff,meta_MDP.std_payoff);
            
            %meta_MDP.payoff_matrix=meta_MDP.samplePayoffMatrix();
            %meta_MDP.outcome_probabilities=meta_MDP.sampleOutcomeProbabilities();
            problem_nr=1+mod(meta_MDP.episode-1,numel(meta_MDP.object_level_MDPs));
            meta_MDP.object_level_MDP=meta_MDP.object_level_MDPs(problem_nr);
            
            meta_MDP.total_nr_clicks=meta_MDP.nr_cells;
            meta_MDP.remaining_nr_clicks=meta_MDP.total_nr_clicks;
            
            [state,meta_MDP]=meta_MDP.sampleS0();
            meta_MDP=meta_MDP.setActions(state);
            meta_MDP.nr_actions=numel(meta_MDP.actions);
            
            state.step=1;
            state.nr_steps=3;
            state.has_plan=false;
            state.s=1;
            state.returns=[];
        end
        
        function [state,meta_MDP]=randomStart(meta_MDP)

            [~,meta_MDP]=meta_MDP.newEpisode();
            [state,meta_MDP]=meta_MDP.sampleS0();
            
            state.step=drawSample(1:state.nr_steps-1);
            state.s=drawSample([meta_MDP.object_level_MDP.states_by_step{state.step}.nr]);
            
            nr_locations=numel(state.observations);
            observed=rand(nr_locations,1)<0.5;

            for l=1:nr_locations
                if observed(l)
                    state.observations(l)=meta_MDP.rewards(l);
                end
            end

            state=meta_MDP.updateBelief(state,find(observed));
        end

        function [state,meta_MDP]=sampleS0(meta_MDP)

            state.step=1;
            state.nr_steps=3;
            state.s=1;            
            
            %sample number of outcomes and number of gambles
            state.observations=NaN(meta_MDP.nr_cells,1);
            state.S=meta_MDP.object_level_MDP.states;
            state.T=meta_MDP.object_level_MDP.T;
            state.A=meta_MDP.object_level_MDP.actions;
            state.available_actions = find(any(squeeze(state.T(state.s,:,:))));
            
            state.nr_steps=meta_MDP.object_level_MDP.horizon;
            state.has_plan=false;
            
            state.returns=[];
            state.plan=[];
            state.decision=NaN;
            state.outcome=NaN;

            
            [meta_MDP.reward_function,meta_MDP.rewards]=meta_MDP.sampleRewards();
            
            meta_MDP=meta_MDP.setActions(state);
            
            %determine the expected value of starting from each cell by
            %solving the meta-level meta_MDP?
            state=meta_MDP.updateBelief(state,1:numel(state.S));           
            
        end
        
    

        function state=updateBelief(meta_MDP,state,new_observations)
            %{ 
                computes state.mu_Q, state.sigma_Q, state.mu_V, and
                state.sigma_V from state.observations and meta_MDP
                mu_Q(s,a): expected return of starting in object-level state s and performing object-level action a according to state.observations and meta_MDP.p_payoffs 
                mu_V(s): expected return of starting in object-level state s and following the optimal object-level policy according to state.observations and meta_MDP.p_payoffs. The expectation is taken with respect to the probability distribution encoded by the meta-level state and the ?optimal policy? maximizes the reward expected according to the probability distribution encoded by the meta-level state
                sigma_Q(s,a), sigma_V(s): uncertainty around the expectations mu_Q(s,a) and mu_V(s).
            %}
            %0. Determine which beliefs have to be updated
            needs_updating=union(new_observations,meta_MDP.getUpStreamStates(new_observations));

            
            %1. Set value of starting from the leave states to zero (Step 3)
            leaf_nodes=meta_MDP.object_level_MDP.states_by_step{state.nr_steps+1};
            for l=1:numel(leaf_nodes)
                state.mu_V(leaf_nodes(l).nr)=0;
                state.sigma_V(leaf_nodes(l).nr)=0;
            end
            
            %2. Propagage the update backwards towards the initial state
            nr_steps=numel(meta_MDP.object_level_MDP.states_by_step);
            for step=(nr_steps-1):-1:1
                
                nodes=meta_MDP.object_level_MDP.states_by_step{step};
                %a) Update belief about state-action values
                for n=1:numel(nodes)
                    node=nodes(n);
                    
                    if not(ismember(node.nr,needs_updating))
                        continue;
                    end
                    
                    for a=1:numel(node.available_actions)
                        action=node.available_actions(a);
                        next_state=meta_MDP.object_level_MDP.states_by_path(num2str([node.path;action]));
                        
                        
                        if isnan(state.observations(node.nr))
                            state.mu_Q(node.nr,action)=meta_MDP.mean_payoff+state.mu_V(next_state.nr);
                            state.sigma_Q(node.nr,action)=sqrt(meta_MDP.std_payoff^2+state.sigma_V(next_state.nr)^2);
                        else
                            state.mu_Q(node.nr,action)=state.observations(node.nr)+state.mu_V(next_state.nr);
                            state.sigma_Q(node.nr,action)=state.sigma_V(next_state.nr);
                        end
                    end
                    
                    %b) Update belief about state value V
                    [state.mu_V(node.nr),state.sigma_V(node.nr)]=...
                        EVofMaxOfGaussians(state.mu_Q(node.nr,node.available_actions),...
                        state.sigma_Q(node.nr,node.available_actions));
                end

            end
                                 
            if any(isnan(state.mu_V))
                disp('Belief is NaN!')
            end
            
        end
        
        function state=updateBeliefOld(meta_MDP,state,new_observations)
            %{ 
                computes state.mu_Q, state.sigma_Q, state.mu_V, and
                state.sigma_V from state.observations and meta_MDP
                mu_Q(s,a): expected return of starting in object-level state s and performing object-level action a according to state.observations and meta_MDP.p_payoffs 
                mu_V(s): expected return of starting in object-level state s and following the optimal object-level policy according to state.observations and meta_MDP.p_payoffs. The expectation is taken with respect to the probability distribution encoded by the meta-level state and the ?optimal policy? maximizes the reward expected according to the probability distribution encoded by the meta-level state
                sigma_Q(s,a), sigma_V(s): uncertainty around the expectations mu_Q(s,a) and mu_V(s).
            %}
            %0. Determine which beliefs have to be updated
            needs_updating=union(new_observations,meta_MDP.getUpStreamStates(new_observations));

            
            %1. Set value of starting from the leave states to zero (Step 3)
            leaf_nodes=meta_MDP.object_level_MDP.states_by_step{state.nr_steps+1};
            for l=1:numel(leaf_nodes)
                state.mu_V(leaf_nodes(l).nr)=0;
                state.sigma_V(leaf_nodes(l).nr)=0;
            end
            
            %2. Set the Q-values of moving to the leave states (Step 2)
            pre_leaf_nodes=meta_MDP.object_level_MDP.states_by_step{state.nr_steps};
            for p=1:numel(pre_leaf_nodes)
                node=pre_leaf_nodes(p);
                
                if not(ismember(node.nr,needs_updating))
                    continue;
                end
                
                for a=1:numel(node.available_actions)
                    action=node.available_actions(a);
                    successor=meta_MDP.object_level_MDP.states_by_path(num2str([node.path;action]));
                    %look up whether the value of the resulting leaf state
                    %has already been observed
                    if isnan(state.observations(successor.nr))
                        %The reward for this transition has not been
                        %observed yet.
                        state.mu_Q(node.nr,action)=meta_MDP.mean_payoff;
                        state.sigma_Q(node.nr,action)=meta_MDP.std_payoff;
                    else 
                        %the reward for this transition has already been
                        %observed
                        state.mu_Q(node.nr,action)=state.observations(successor.nr);
                        state.sigma_Q(node.nr,action)=0;
                    end
                end
            end
            
            %3. Set the values of of the state before the leaf (Step 2)
            for p=1:numel(pre_leaf_nodes)
                node=pre_leaf_nodes(p);

                if not(ismember(node.nr,needs_updating))
                    continue;
                end
                
                observed_rewards=NaN(numel(node.available_actions),1);
                for a=1:numel(node.available_actions)
                    action=node.available_actions(a);
                    successors(a)=meta_MDP.object_level_MDP.states_by_path(num2str([node.path;action]));
                    observed_rewards(a)=state.observations(successors(a).nr);                                        
                end
                
                nr_unobserved_successors=sum(isnan(observed_rewards));
                
                if nr_unobserved_successors==0 %none of the rewards has been observed
                    state.mu_V(node.nr)=max(observed_rewards);
                    state.sigma_V(node.nr)=0;
                    
                elseif nr_unobserved_successors==numel(observed_rewards) %all of the rewards have been observed

                    mu=repmat(meta_MDP.mean_payoff,[nr_unobserved_successors,1]);
                    sigma=repmat(meta_MDP.std_payoff,[nr_unobserved_successors,1]);
                   
                    [state.mu_V(node.nr),state.sigma_V(node.nr)]=...
                        EVofMaxOfGaussians(mu,sigma);
                else %the reward is known for some actions but not for others
                    max_known_reward=max(observed_rewards);
                    
                    mu=[max_known_reward;repmat(meta_MDP.mean_payoff,[nr_unobserved_successors,1])];
                    sigma=[0;repmat(meta_MDP.std_payoff,[nr_unobserved_successors,1])];
                    [state.mu_V(node.nr),state.sigma_V(node.nr)]=...
                        EVofMaxOfGaussians(mu,sigma);
                end
            end
            
            %4. Set the Q-values of moves leading to the pre-leaf states
            hallway_nodes=meta_MDP.object_level_MDP.states_by_step{2};
            for n=1:numel(hallway_nodes)
                node=hallway_nodes(n);

                if not(ismember(node.nr,needs_updating))
                    continue;
                end                
                
                for a=1:numel(node.available_actions)
                    action=node.available_actions(a);
                    next_state=meta_MDP.object_level_MDP.states_by_path(num2str([node.path;action]));
                    
                    
                    if isnan(state.observations(next_state.nr))
                        state.mu_Q(node.nr,action)=meta_MDP.mean_payoff+state.mu_V(next_state.nr);
                        state.sigma_Q(node.nr,action)=sqrt(meta_MDP.std_payoff^2+state.sigma_V(next_state.nr)^2);
                    else
                        state.mu_Q(node.nr,action)=state.observations(next_state.nr)+state.mu_V(next_state.nr);
                        state.sigma_Q(node.nr,action)=state.sigma_V(next_state.nr);
                    end
                end
                %5. Set the value of the states reached by the first move    
                [state.mu_V(node.nr),state.sigma_V(node.nr)]=...
                    EVofMaxOfGaussians(state.mu_Q(node.nr,node.available_actions),...
                    state.sigma_Q(node.nr,node.available_actions));
                
            end
                        
                        
            %6. Set the Q-value of moving into each state on the second
            %layer (Step 1) to expected immediate reward plus value of the
            %successor state.
            start_state=meta_MDP.object_level_MDP.states_by_step{1};
            for a=1:numel(start_state.available_actions)
                action=start_state.available_actions(a);
                next_state=meta_MDP.object_level_MDP.states_by_path(num2str([start_state.path;action]));
                
                if not(ismember(next_state.nr,needs_updating))
                    continue;
                end
                
                if isnan(state.observations(next_state.nr))
                    state.mu_Q(start_state.nr,action)=meta_MDP.mean_payoff+state.mu_V(next_state.nr);
                    state.sigma_Q(start_state.nr,action)=sqrt(meta_MDP.std_payoff^2+state.sigma_V(next_state.nr)^2);
                else
                    state.mu_Q(start_state.nr,action)=state.observations(next_state.nr)+state.mu_V(next_state.nr);
                    state.sigma_Q(start_state.nr,action)=state.sigma_V(next_state.nr);
                end
            end
            
            
            %7. Set the value of of the initial state (Step 1).
            [state.mu_V(start_state.nr),state.sigma_V(start_state.nr)]=...
                EVofMaxOfGaussians(state.mu_Q(start_state.nr,:),...
                state.sigma_Q(start_state.nr,:));
            
            if any(isnan(state.mu_V))
                disp('Belief is NaN!')
            end
            
        end        
        
        function terminate=isTerminalState(meta_MDP,state)
            terminate=state.step>state.nr_steps;
        end
        
        function ER=expectedReward(meta_MDP,state,move)
            
            next_location=meta_MDP.object_level_MDP.nextState(state.s,move);
            
            if isnan(state.observations(next_location))
                ER=meta_MDP.mean_payoff;
            else
                ER=state.observations(next_location);
            end
            
        end
   
        function [E_regret_reduction,meta_MDP]=expectedRegretReduction(meta_MDP,s_old,c)
            
            
            [next_states,p_next_states,meta_MDP]=meta_MDP.predictNextState(s_old,c);

            regret_reduction=meta_MDP.regretReduction(s_old,next_states,s_old.s);
            
            E_regret_reduction=dot(p_next_states,regret_reduction);
            
            %regret_reduction_minus_cost=E_regret_reduction-meta_MDP.time_per_click*meta_MDP.time_cost_per_sec;
        end

        
        function [r,next_state,PR]=simulateTransition(meta_MDP,state,c)
            
            start_state=state;
            
            if c.is_computation
                if isnan(state.observations(c.state)) %not observed yet
                    
                    state.observations(c.state)=meta_MDP.object_level_MDP.rewards(c.from_state,c.state,c.move);%meta_MDP.object_level_MDP.rewards(c.from_state,c.state,c.move);
                    next_state=meta_MDP.updateBelief(state,c.state);
                    
                else %the box had already been opened before
                    next_state=state;
                end
                
                r=-meta_MDP.cost_per_click;
                next_state.outcome=NaN;
                next_state.returns=[next_state.returns;r];
            elseif c.is_decision_mechanism
                
                [state,decision]=meta_MDP.decide(state,c);
                c.decision=decision;

                next_state=state;
                next_state.s=meta_MDP.object_level_MDP.nextState(state.s,...
                    c.decision);                
                next_state.available_actions = find(any(squeeze(next_state.T(next_state.s,:,:))));
                
                r=meta_MDP.rewards(next_state.s);%meta_MDP.object_level_MDP.rewards(state.s,next_state.s,c.decision)-...
                  %  meta_MDP.costOfPlanning(state,c);
                                
                if isnan(next_state.s)
                    throw(MException('invalidValue:isNaN','next state not found'))
                end
                
                next_state.step=state.step+1;
                next_state.returns=[next_state.returns; r];
                %{
                %information state after the choice but before the next
                episode
                next_state.decision=action.gamble;
                next_state.outcome=o;
                next_state.observations(o,action.gamble)=r;
                next_state.mu(action.gamble)=meta_MDP.reward_function(next_state.outcome,action.gamble);
                next_state.sigma(action.gamble)=0;
                
                %gambles for which the sampled outcome has been observed
                next_state.mu(not(isnan(next_state.observations(o,:))))=meta_MDP.reward_function(o,not(isnan(next_state.observations(o,:))));
                next_state.sigma(not(isnan(next_state.observations(o,:))))=zeros(sum(not(isnan(next_state.observations(o,:)))),1);
                
                %gambles for which the sampled outcome has not been
                %observed
                next_state.mu(isnan(next_state.observations(o,:)))=zeros(sum(isnan(next_state.observations(o,:))),1);
                next_state.sigma(isnan(next_state.observations(o,:)))=meta_MDP.std_payoff;
                %}
                %information state at the beginning of the next episode
                %if meta_MDP.isTerminalState(next_state)
                %    [next_state,meta_MDP]=meta_MDP.newEpisode();
                %end
                                
                %end the episode
            end
            
            if meta_MDP.add_pseudorewards
                if strcmp(meta_MDP.pseudoreward_type,'myopicVOC')
                    PR=meta_MDP.myopicVOC(state,c);
                elseif strcmp(meta_MDP.pseudoreward_type,'regretReduction')
                    PR=meta_MDP.regretReduction(state,next_state);
                elseif strcmp(meta_MDP.pseudoreward_type,'featureBased')
                    PR = meta_MDP.featureBasedPR(start_state,c);
                end
            else
                PR=0;
            end
                        
        end
        
        function [state,decision]=decide(meta_MDP,state,c)
            %determine a move according to the current belief state
            
            location=meta_MDP.object_level_MDP.states(state.s);
            [ER,d_index]=max(state.mu_Q(state.s,location.available_actions));
            decision=location.available_actions(d_index);
            
            %{
            if c.planning_horizon==0
                if not(state.has_plan)
                    location=meta_MDP.object_level_MDP.states(state.s);
                    available_moves=location.available_actions;
                    c.decision=drawSample(available_moves);
                else
                    c.decision=state.plan(state.s,state.step_in_plan);
                    state.step_in_plan=state.step_in_plan+1;
                    
                    if state.step_in_plan>state.planned_nr_steps
                        state.has_plan=false;
                        state.planned_nr_steps=0;
                    end
                    
                end
            elseif c.planning_horizon>0
                
                nr_locations=numel(state.S);
                nr_actions=numel(meta_MDP.object_level_MDP.actions);
                R_hat=-realmax*ones(nr_locations,nr_locations,nr_actions); %impossible actions will be assigned a reward of -inf
                T_tilde=nan(size(meta_MDP.object_level_MDP.T));
                
                for from=1:nr_locations
                    
                    available_actions=state.S(from).available_actions;
                    
                    if isempty(available_actions)
                        T_tilde(from,:,:)=repmat(deltaDistribution(from,1:nr_locations),[1,1,nr_actions]);
                        R_hat(from,:,:)=zeros(nr_locations,nr_actions);
                    end
                    
                    for a=1:numel(available_actions)
                        
                        action=available_actions(a);
                        T_tilde(from,:,action)=meta_MDP.object_level_MDP.T(from,:,action);
                        to=meta_MDP.object_level_MDP.nextState(from,action);
                        
                        if isnan(state.observations(to))
                            R_hat(from,to,action)=meta_MDP.mean_payoff;
                        else
                            R_hat(from,to,action)=state.observations(to);
                        end
                    end
                end
                                
                [temp1, state.plan, temp2] = mdp_finite_horizon(T_tilde, R_hat, 1, c.planning_horizon);
                state.has_plan=true;
                state.step_in_plan=1;
                state.planned_nr_steps=c.planning_horizon;
                c.decision=state.plan(state.s,state.step_in_plan);
                
                if not(ismember(c.decision,state.S(state.s).available_actions))
                    save Error.mat
                    throw(MException('isNaN','The planned action is not available in the current state.'))
                end
                
                state.step_in_plan=state.step_in_plan+1;
                if state.step_in_plan>state.planned_nr_steps
                    state.has_plan=false;
                    state.planned_nr_steps=0;
                end
            end
            %}
            
        end
        
        function cost_of_planning=costOfPlanning(meta_MDP,state,c)
            %compute the number and length of all paths from the current 
            location=meta_MDP.object_level_MDP.states(state.s);
            moves=location.available_actions;
            
            cost_of_planning=0;
            if c.planning_horizon>0
                for m=1:numel(moves)
                    move=moves(m);
                    next_state=state; next_state.s=meta_MDP.object_level_MDP.nextState(state.s,move);
                    next_c=c; next_c.planning_horizon=c.planning_horizon-1;
                    cost_of_planning=cost_of_planning+...
                        meta_MDP.cost_per_operation+meta_MDP.costOfPlanning(next_state,next_c);
                end                
            end
            
            %cost_of_planning=nr_planning_operations*meta_MDP.cost_per_operation;
        end
        
        function [next_states,p_next_states,meta_MDP]=predictNextState(meta_MDP,state,c)
            
            if and(structsEqual(state,meta_MDP.query_state),...
                   structsEqual(c,meta_MDP.query_computation))
               
               next_states=meta_MDP.next_states;
               p_next_states=meta_MDP.p_next_states;
               
               return
            end
            
            if c.is_decision_mechanism
                [next_state,decision]=meta_MDP.decide(state,c);
                next_state.decision=decision;
                
                next_location=meta_MDP.object_level_MDP.nextState(state.s,decision);
                next_state.s=next_location;
                r_meta=meta_MDP.expectedReward(state,decision)+...
                    meta_MDP.costOfPlanning(state,c);
                next_state.returns=[next_state.returns; r_meta];
                
                if not(isnan(state.observations(next_location)))
                    next_states=[next_state];
                    p_next_states=[1];
                else
                    next_states=repmat(next_state,[meta_MDP.nr_possible_payoffs,1]);
                    %inspect another payoff
                    for v=1:meta_MDP.nr_possible_payoffs
                        next_states(v).observations(next_state.s)=...
                            meta_MDP.payoff_values(v);
                        next_states(v)=meta_MDP.updateBelief(next_states(v));
                        next_states(v).step=state.step+1;
                    end
                    p_next_states=meta_MDP.p_payoff_values;                    
                end
                
            elseif c.is_computation
                
                if isnan(state.observations(c.state))
                    next_states=repmat(state,[meta_MDP.nr_possible_payoffs,1]);
                    %inspect another payoff
                    for v=1:meta_MDP.nr_possible_payoffs
                        next_states(v).observations(c.state)=...
                            meta_MDP.payoff_values(v);
                        next_states(v)=meta_MDP.updateBelief(next_states(v),c.state);
                        next_states(v).returns=[next_states(v).returns; -meta_MDP.cost_per_click];
                    end
                    p_next_states=meta_MDP.p_payoff_values;
                else
                    next_states=repmat(state,[1,1]);                    
                    p_next_states=1;
                end
                
            end
            
            meta_MDP.next_states=next_states;
            meta_MDP.p_next_states=p_next_states;
            meta_MDP.query_state=state;
            meta_MDP.query_computation=c;
            
        end
        
        function [payoffs,rewards]=sampleRewards(meta_MDP)
            %sample values for each cell of the payoff matrix
            
            locations=meta_MDP.object_level_MDP.states;
            nr_locations=numel(locations);
            moves=meta_MDP.object_level_MDP.actions;
            nr_moves=numel(moves);
            
            payoffs=NaN(nr_locations,nr_moves);
            rewards=zeros(nr_locations,1);
            
            rewards(1)=0;
            for l=1:numel(locations)
                for m=1:nr_moves
                    
                    if ismember(m,locations(l).available_actions)
                        v=meta_MDP.mean_payoff+meta_MDP.std_payoff*randn();
                        payoffs(l,m)=max(meta_MDP.min_payoff,min(meta_MDP.max_payoff,round(100*v)/100));
                        next_state=meta_MDP.object_level_MDP.nextState(locations(l).nr,m);
                        rewards(next_state)=payoffs(l,m);
                    else
                        payoffs(l,m)=NaN;
                    end
                end
            end
            
            
        end
                
        function [actions,meta_MDP]=getActions(meta_MDP,state)
            meta_MDP=meta_MDP.setActions(state);
            actions=meta_MDP.actions;
        end
        
        function [meta_MDP,actions]=setActions(meta_MDP,state)
            
            nr_decision_mechanisms=state.nr_steps-state.step+2; %plan 0,...,nr. remaining steps [h=0 can mean either plan execution or random choice depending on whether the agent currently has a plan]
            nr_acquisitions=sum(not(isnan(meta_MDP.nr_cells)));            
            nr_actions=nr_decision_mechanisms+nr_acquisitions;
            
            meta_MDP.actions=repmat(struct('is_decision_mechanism',true,...
                'is_computation',false,'planning_horizon',0,'decision',0,...
                'from_state',0,'move',0,'state',0),[nr_actions,1]);
            
            %decision mechanisms
            for h=1:nr_decision_mechanisms
                meta_MDP.actions(h).is_decision_mechanism=true;
                meta_MDP.actions(h).is_computation=false;
                meta_MDP.actions(h).planning_horizon=h-1;                
            end
            
            %inspect additional rewards
            a=nr_decision_mechanisms;
            from_states=meta_MDP.object_level_MDP.states;
            for s=1:numel(from_states)
                from_state=from_states(s);
                actions=from_states(s).available_actions;
                for i=1:numel(actions)
                    action=actions(i);
                    next_state=meta_MDP.object_level_MDP.nextState(from_state.nr,action);
                    if isnan(state.observations(next_state)) %If this cell has not been observed yet, then allow it to be observed.
                        a=a+1;
                        meta_MDP.actions(a).is_decision_mechanism=false;
                        meta_MDP.actions(a).is_computation=true;
                        meta_MDP.actions(a).from_state=from_state.nr;
                        meta_MDP.actions(a).move=action;
                        meta_MDP.actions(a).state=next_state;
                    end
                end
            end
            
            meta_MDP.nr_actions=numel(meta_MDP.actions);
            actions=meta_MDP.actions;
            
        end                
        
        function PR=getPseudoRewards(meta_MDP,state)
            
            a=0;
            for o=1:meta_MDP.nr_outcomes
                for g=1:meta_MDP.nr_gambles
                    a=a+1;
                    action.is_decision=false;
                    action.outcome=o;
                    action.gamble=g;
                    
                    PR(o,g)=meta_MDP.myopicVOC(state,action);
                    
                end
            end
            
        end
        
        function next_state=addObservation(meta_MDP,state,outcome,gamble,payoff)
            
            next_state=state;
            
            %update observations
            next_state.observations(outcome,gamble)=payoff;
            
            %update the remaining nr of clicks
            next_state.remaining_nr_clicks=state.remaining_nr_clicks-1;
            
            %update belief state
            unobserved=isnan(state.observations);
            
            if unobserved(outcome,gamble)
                next_state.mu(gamble)=state.mu(gamble)+...
                    meta_MDP.outcome_probabilities(outcome)*payoff;
                
                unobserved_outcomes=isnan(next_state.observations(:,gamble));
                next_state.sigma(gamble)=meta_MDP.std_payoff*...
                    sqrt(sum(meta_MDP.outcome_probabilities(unobserved_outcomes).^2));
            end
            
        end
        
        function state_action_features=extractStateActionFeatures(meta_MDP,state,action)
            %state_features=meta_MDP.extractStateFeatures(state);
            action_features=meta_MDP.extractActionFeatures(state,action);
            
            %state_action_features=[state_features;action_features];
            state_action_features=action_features;
            
            if any(isnan(state_action_features))
                throw(MException('MException:isNaN','features are NaN'))
            end
        end
        
        function state_features=extractStateFeatures(meta_MDP,states)
            
            %find the best action in the current state            
            for i=1:numel(states)
                state=states(i);
                
                location=state.s;
                
                a_star=argmax(state.mu_Q(location,:));                
                [~,b]=kthLargestElement(state.mu_Q(location,:),2);
                
                [E_max,sigma_max]=EVofMaxOfGaussians(state.mu_Q(location,:),...
                    state.sigma_Q(location,:));
                
                %{
                myopic_VOC=zeros(meta_MDP.nr_outcomes,meta_MDP.nr_gambles);
                for g=1:meta_MDP.nr_gambles
                    for o=1:meta_MDP.nr_outcomes
                        action.gamble=g; action.outcome=o;
                        action.is_decision=false;
                        myopic_VOC(o,g)=meta_MDP.myopicVOC(state,action);
                    end
                end
                %}

                expected_regret=meta_MDP.expectedRegret(state.mu_Q(location,:),...
                    state.sigma_Q(location,:));                                
                
                state_features(:,i)=[state.mu_Q(location,a_star);state.sigma_Q(location,a_star);...
                    E_max; sigma_max;...
                    state.mu_Q(location,b); state.sigma_Q(location,b);...
                    %max(state.sigma(:));
                    %;size(state.observations,2)
                    ];
                %; max(myopic_VOC(:))
            end
        end
        
        function action_features=extractActionFeatures(meta_MDP,state,c)
                        
            if c.is_decision_mechanism
                
                %{
                [state,decision]=meta_MDP.decide(state,c);
                location=state.s;
                
                E_R=state.mu_Q(location,decision);
                sigma_R=state.sigma_Q(location,decision);

                mu=state.mu_Q(location,:);
                sigma=state.sigma_Q(location,:);
                [E_max,~]=EVofMaxOfGaussians(mu,sigma);

                expected_regret=E_max-E_R;
                
                [p_argmax,sigma_argmax]=distributionOfArgMax(mu,sigma);
                sigma_best_action=sigma_argmax(decision);
                p_best_action=p_argmax(decision);
                
                planning_horizon=c.planning_horizon;
                
                remaining_steps=state.nr_steps-state.step+1;
                underplanning=max(0,remaining_steps-c.planning_horizon);
                
                complete_planning=c.planning_horizon==remaining_steps;

                regret_reduction=0;
                
                cost=meta_MDP.costOfPlanning(state,c);
                uncertainty_reduction=0;
                                
                %}
                VOC=0;
                %nr_cells_uninspected=sum(isnan(state.observations(:)));
                
                VPI=0;
            else
                %{
                expected_regret=0;
                [regret_reduction,meta_MDP]=expectedRegretReduction(meta_MDP,state,c);
                
                [uncertainty_reduction,meta_MDP]=meta_MDP.expectedUncertaintyReduction(state,c);
                sigma_R=0;
                p_best_action=0;
                sigma_best_action=0;
                underplanning=0;
                complete_planning=0;
                cost=meta_MDP.cost_per_click;
                %}
                VPI=meta_MDP.computeVPI(state,c);                
                [VOC,meta_MDP]=meta_MDP.myopicVOC(state,c);
            end
            
            ER_act=max(state.mu_Q(state.s,:));
            
            %{
            action_features=[expected_regret;regret_reduction;VOC;...
                uncertainty_reduction; sigma_R;p_best_action;sigma_best_action;...
                underplanning;complete_planning;cost];
            %}
            action_features=[VPI; VOC; ER_act];
        end
        
        function VPI=computeVPI(meta_MDP,state,c)
            
            if c.is_computation
                
                corresponding_action=meta_MDP.getCorrespondingAction(state,c);
                
                if or(isnan(corresponding_action),not(isnan(state.observations(c.state))))
                    VPI=0;
                else
                    VPI=valueOfPerfectInformation(state.mu_Q(state.s,:),state.sigma_Q(state.s,:),corresponding_action);
                end
            else
                VPI=0;
            end
            
        end
        
        function corresponding_action=getCorrespondingAction(meta_MDP,state,c)
            
            [~,downstream_states_by_action]=meta_MDP.getDownStreamStates(state);
            corresponding_action=NaN;
            for a=1:numel(downstream_states_by_action)
                if ismember(c.state,downstream_states_by_action{a})
                    corresponding_action=a;
                end
            end
        end
        
        function V_deliberate=VfullDeliberation(meta_MDP,state)
            location=state.s;
            c.planning_horizon=state.nr_steps-state.step;
            planning_cost=meta_MDP.costOfPlanning(state,c);
                        
            downstream=meta_MDP.getdownstreamStates(state);
            information_cost=meta_MDP.cost_per_click*...
                sum(isnan(state.observations(downstream)));
            
            V_deliberate=state.mu_V(location)-planning_cost-information_cost;
        end
        
        function [E_uncertainty_reduction,meta_MDP]=expectedUncertaintyReduction(meta_MDP,state,c)
            
            [next_states,p_next_states,meta_MDP]=meta_MDP.predictNextState(state,c);

            uncertainty_reduction=zeros(size(next_states));
            
            for n=1:numel(next_states)
                uncertainty_reduction(n)=sum(state.sigma_Q(state.s,:)-...
                    next_states(n).sigma_Q(state.s,:));
            end
            
            E_uncertainty_reduction=dot(p_next_states,uncertainty_reduction);
                                    
        end
        
        function [downstream,downstream_by_action]=getDownStreamStates(meta_MDP,state)
                downstream=[];
                downstream_by_action=cell(4,1);
                
                states=state.S;
                current_path=states(state.s).path;
                steps=1:numel(current_path);
                
                if isempty(current_path)
                    downstream=1:numel(states);

                    for s=2:numel(states)
                        
                        downstream_by_action{states(s).path(1)}=...
                            [downstream_by_action{states(s).path(1)},s];                        
                    end
                    
                else                    
                    for s=1:numel(states)
                        if length(states(s).path)>length(current_path)
                            if all(states(s).path(steps)==current_path)
                                downstream=[downstream; s];
                                
                                downstream_by_action{states(s).path(length(current_path)+1)}=...
                                    [downstream_by_action{states(s).path(length(current_path)+1)},s];
                                
                            end
                        end
                    end
                end
                
        end
        
        function [PR_meta,PR_meta_integrated,meta_MDP]=PRofFullDeliberationPolicy(meta_MDP,state,c)
            V_from=meta_MDP.VfullDeliberation(state);
            
            [next_states, p_next_states,meta_MDP]=meta_MDP.predictNextState(state,c);
            
            V_to=zeros(numel(next_states),1);
            for n=1:numel(next_states)
                V_to(n)=meta_MDP.VfullDeliberation(next_states(n));
                R(n)=next_states(n).returns(end);
            end
            
            E_V_to=dot(p_next_states(:),V_to(:));
            
            PR_meta=E_V_to-V_from;
            
            E_R=dot(p_next_states,R);
            PR_meta_integrated=PR_meta+E_R;
        end
        
        function expected_regret=expectedRegret(meta_MDP,mu,sigma)
            max_mu=max(mu); %if >1 mu's are max, shouldn't a_star be selected randomly (not the first)
            
            [E_max,~]=EVofMaxOfGaussians(mu,sigma);
                        
            expected_regret=E_max-max_mu;
        end
        
        function [VOC,meta_MDP]=myopicVOC(meta_MDP,state,c)
                        
             if c.is_computation  
                if and(isnan(state.observations(c.state)),c.state>1)
                    
                    if ismember(c.state,meta_MDP.getDownStreamStates(state))
                        location=state.s;
                        %actions=meta_MDP.object_level_MDP.actions_by_state{c.state};
                        path=meta_MDP.object_level_MDP.states(c.state).path;
                        
                        a=path(state.step);                        
                        mu_prior=state.mu_Q(state.s,:);
                        
                        
                        %if hallway state
                        if ismember(c.state,meta_MDP.object_level_MDP.hallway_states)
                            VOC=myopicVOCAdditive(meta_MDP,mu_prior,a);
                        end
                                                
                        if ismember(c.state,meta_MDP.object_level_MDP.leafs)
                            
                            parent=meta_MDP.object_level_MDP.parent_by_state{c.state};
                            siblings=setdiff([state.S(parent).actions.state],c.state);
                            
                            if any(isnan(state.observations(siblings)))
                                %if leaf node with unknown sibling(s)
                                VOC=myopicVOCMaxUnknown(meta_MDP,mu_prior,a);
                            else
                                %if leaf node with known sibling(s)
                                alternative=max(state.observations(siblings));
                                VOC=myopicVOCMaxKnown(meta_MDP,mu_prior,a,alternative);
                            end
                        end

                    else
                        VOC=0-meta_MDP.cost_per_click;
                    end
                    
                    %{
                    [~,a_old]=max(state.mu_Q(location,:));
                    
                    [next_states,p_next_states,meta_MDP]=meta_MDP.predictNextState(state,c);
                    
                    delta_EV=zeros(numel(next_states),1);
                    for s=1:numel(next_states)
                        max_posterior_EV=max(next_states(s).mu_Q(location,:));
                        delta_EV(s)=max_posterior_EV-next_states(s).mu_Q(location,a_old);
                    end
                    
                    VOC=dot(p_next_states,delta_EV)-meta_MDP.cost_per_click;
                    %}
                else
                    VOC=0-meta_MDP.cost_per_click;
                end
            elseif c.is_decision_mechanism
                VOC=0;
            end
        end
        
        function VOC=myopicVOCAdditive(meta_MDP,mu_prior,a)
            %This function evaluates the VOC of inspecting a hallway cell
            %downstream of the current state.
            %mu_prior: prior means of returns of available actions
            %sigma_prior: prior uncertainty of returns of available actions
            %a: action about which more information is being collected
            
            [mu_sorted,pos_sorted]=sort(mu_prior,'descend');
            
            mu_alpha=mu_sorted(1);
            alpha=pos_sorted(1);
            
            mu_beta=mu_sorted(2);
            beta=pos_sorted(2);
            
            if a==alpha
                %information is valuable if it reveals that action c is suboptimal
                
                %To change the decision, the sampled value would have to be less than
                %ub
                ub=mu_beta+meta_MDP.mean_payoff-mu_alpha;                
                VOC=meta_MDP.std_payoff^2*normpdf(ub,meta_MDP.mean_payoff,meta_MDP.std_payoff)-...
                    (mu_alpha-mu_beta)*normcdf(ub,meta_MDP.mean_payoff,meta_MDP.std_payoff)-...
                    meta_MDP.cost_per_click;
                
            else
                %information is valuable if it reveals that action is optimal
                
                %To change the decision, the sampled value would have to be larger than
                %lb
                lb=mu_alpha+meta_MDP.mean_payoff-mu_prior(a);                
                VOC=meta_MDP.std_payoff^2*normpdf(lb,meta_MDP.mean_payoff,meta_MDP.std_payoff)-...
                    (mu_alpha-mu_prior(a))*(1-normcdf(lb,meta_MDP.mean_payoff,meta_MDP.std_payoff))-...
                    meta_MDP.cost_per_click;
            end
        end

        function VOC=myopicVOCMaxKnown(meta_MDP,mu_prior,a,known_alternative)
            %This function evaluates the VOC of inspecting a leaf cell
            %downstream of the current state where the values of the other leaf(s) is/are known.
            %mu_prior: prior means of returns of available actions            
            %a: action about which more information is being collected
            %known_alternative: maximum of the other known leafs
            [mu_sorted,pos_sorted]=sort(mu_prior,'descend');
            
            mu_alpha=mu_sorted(1);
            alpha=pos_sorted(1);
            
            mu_beta=mu_sorted(2);
            beta=pos_sorted(2);
            
            E_max=EVofMaxOfGaussians([meta_MDP.mean_payoff,known_alternative],...
                [meta_MDP.std_payoff,0]);
            
            if a==alpha
                %information is valuable if it reveals that action c is suboptimal
                
                %The decision can only change if E[max
                %{known_alternative,x}]-k>mu_alpha-mu_beta
                
                if (E_max-known_alternative<= mu_alpha-mu_beta)
                    VOC=0-meta_MDP.cost_per_click;
                else
                    %to change the decision x would have to be less than
                    %the known alternative
                    ub=known_alternative;                    
                    VOC=normcdf(ub,meta_MDP.mean_payoff,meta_MDP.std_payoff)*...
                        (mu_beta-(mu_alpha-E_max+known_alternative))-meta_MDP.cost_per_click;                    
                end                                                
                
            else
                %information is valuable if it reveals that action is optimal                
                
                %To change the decision, the sampled value would have to be larger than
                %lb
                lb=mu_alpha-mu_prior(a)+E_max;
                
                VOC=meta_MDP.std_payoff^2*normpdf(lb,meta_MDP.mean_payoff,meta_MDP.std_payoff)-...
                    (mu_alpha-mu_prior(a)-E_max-meta_MDP.mean_payoff)*...
                    (1-normcdf(lb,meta_MDP.mean_payoff,meta_MDP.std_payoff))-...
                    meta_MDP.cost_per_click;
            end
            
        end        

        function VOC=myopicVOCMaxUnknown(meta_MDP,mu_prior,a)
            %This function evaluates the VOC of inspecting a leaf cell
            %downstream of the current state where the values of the other leaf(s) is/are known.
            %mu_prior: prior means of returns of available actions            
            %a: action about which more information is being collected
            %known_alternative: maximum of the other known leafs
            [mu_sorted,pos_sorted]=sort(mu_prior,'descend');
            
            mu_alpha=mu_sorted(1);
            alpha=pos_sorted(1);
            
            mu_beta=mu_sorted(2);
            beta=pos_sorted(2);
            
            E_max=EVofMaxOfGaussians([meta_MDP.mean_payoff,meta_MDP.mean_payoff],...
                [meta_MDP.std_payoff,meta_MDP.std_payoff]);
            
            if a==alpha
                %information is valuable if it reveals that action c is suboptimal
                
                lb=meta_MDP.mean_payoff-3*meta_MDP.std_payoff;
                ub=meta_MDP.mean_payoff;
                
                VOC=integral(@(x) normpdf(x,meta_MDP.mean_payoff,meta_MDP.std_payoff).*...
                    max(0, mu_beta - (mu_alpha-E_max+ETruncatedNormal(meta_MDP.mean_payoff,meta_MDP.std_payoff,...
                    x,inf))),lb,ub)-meta_MDP.cost_per_click;                
            else
                %information is valuable if it reveals that action is optimal                
                lb=meta_MDP.mean_payoff;
                ub=meta_MDP.mean_payoff+3*meta_MDP.std_payoff;
                
                VOC=integral(@(x) normpdf(x,meta_MDP.mean_payoff,meta_MDP.std_payoff).*...
                    max(0, (mu_prior(a)-E_max+ETruncatedNormal(meta_MDP.mean_payoff,meta_MDP.std_payoff,...
                    x,inf))-mu_alpha),lb,ub)-meta_MDP.cost_per_click;                                
            end
            
        end                
        
        function regret_reduction=regretReduction(meta_MDP,s_old,s_new,object_level_state)
            %pseudo-reward derived from the potential function
            %phi(s)=E[max EV]-expected_regret
            mu_old=s_old.mu_Q(object_level_state,:);
            sigma_old=s_old.sigma_Q(object_level_state,:);
            
            
            expected_regret_old_state=meta_MDP.expectedRegret(mu_old,sigma_old);
            Phi_old=-expected_regret_old_state;
            
            for s=1:numel(s_new)
                next_state=s_new(s);
                mu_new=next_state.mu_Q(object_level_state,:);
                sigma_new=next_state.sigma_Q(object_level_state,:);    
                expected_regret_new_state=meta_MDP.expectedRegret(mu_new,sigma_new);
                
                Phi_new=-expected_regret_new_state;
                regret_reduction(s)=Phi_new-Phi_old;
            end
            %Phi_new=expected_maximum_new-expected_regret_new_state;
            %Phi_old=expected_maximum_old-expected_regret_old_state;

        end
        
        function upstream=getUpStreamStates(meta_MDP,locations)
            upstream=[];
            
            if numel(locations)==1
                locations(1)=locations;
            end
            
            states=meta_MDP.object_level_MDP.states;
            
            for l=1:numel(locations)
                current_path=states(locations(l)).path;
                
                if isempty(current_path)
                    upstream=[];
                else
                    upstream=[1];
                    for s=2:numel(states)
                        steps=1:numel(states(s).path);
                        if length(current_path)>length(states(s).path)
                            if all(states(s).path==current_path(steps))
                                upstream=[upstream; s];
                            end
                        end
                    end
                end
            end
            
        end
        
        function selected_computations=piBlinkered(meta_MDP,state)
            %c_blinkered is the computation the blinkered policy would choose in
            %state s
            computations = meta_MDP.getActions(state);            
            
            Q_blinkered=NaN(numel(computations),1);
            for c=1:numel(computations)
                
                computation=computations(c);
                
                if computation.is_computation                 
                    if computation.state==state.s
                        Q_blinkered(c)=-meta_MDP.cost_per_click;
                    else
                        
                        dummy_state=state;
                        dummy_state.s=computation.state;                        
                        upstream=meta_MDP.getUpStreamStates([computation.state]);
                        downstream=meta_MDP.getDownStreamStates(dummy_state);
                        arm_states=setdiff(union(union(upstream,downstream),computation.state),1);
                        observed = not(isnan(state.observations(arm_states)));
                        corresponding_action = meta_MDP.getCorrespondingAction(state,computation); 
                        
                        if isnan(corresponding_action)%this computation does not pertain to any of the still available object-level actions
                            Q_blinkered(c)=-inf;
                        else
                            
                            sigma_mu = state.sigma_Q(state.s,corresponding_action);
                            E_mu = state.mu_Q(state.s,corresponding_action);
                            
                            best_alternative = max(state.mu_Q(state.s,setdiff(1:4,corresponding_action)));
                            delta_mu = E_mu - best_alternative;
                            
                            if computation.is_decision_mechanism
                                c_blinkered=1;
                            else
                                c_blinkered=1+find(arm_states==computation.state);
                            end
                            
                            Q_blinkered(c)=getQBlinkered(meta_MDP,delta_mu,sigma_mu,observed,c_blinkered)+E_mu;
                        end
                    end
                else
                    [Q_blinkered(c),computation.move]=max(state.mu_Q(state.s,:));
                end
            end
            
            c_max = computations(argmax(Q_blinkered(:)));
            
            selected_computations = [c_max];
            if (c_max.is_computation)
                if and(ismember(c_max.state,meta_MDP.object_level_MDP.leafs),mod(c_max.state,2)==1)
                    second_computation = c_max;
                    parent_state=meta_MDP.object_level_MDP.parent_by_state{c_max.state};
                    sibling=setdiff([meta_MDP.object_level_MDP.states(parent_state).actions.state],c_max.state);                    
                    second_computation.state=sibling;
                    
                    selected_computations = [selected_computations, second_computation];
                end
            end
        end
        
        function Q_hat=getQBlinkered(meta_MDP,delta_mu,sigma_mu,observed,c)
            %Q_hat is an approximation to the blinkered Q-function when the
            %current difference between the mean of the considered arm and
            %the best alternative is delta_mu and the STD of the mean of
            %the considered arm is sigma_mu. observed(i)=1 if the i-th
            %cell of the considered arm has already been observed.
            states=meta_MDP.states_blinkered;
            
            obs_id = bi2de(observed(:)')+1;
            state_nr = @(observation_vector,mu,sigma) find(and(and(...
                states.mu(:)==mu,states.sigma(:)==sigma),...
                states.observation_id(:)==obs_id));
            
            delta_mu_values = unique(states.mu(:));
            sigma_mu_values = unique(states.sigma(:));
            
            [~,mu_id] = min(abs(delta_mu-delta_mu_values));
            [~,sigma_id]  = min(abs(sigma_mu-sigma_mu_values));
            mu=delta_mu_values(mu_id);
            sigma=sigma_mu_values(sigma_id);
            
            
            Q_hat = meta_MDP.Q_blinkered(state_nr(observed,mu,sigma),c);            
            
        end
        
        function meta_MDP=computeBlinkeredPolicy(meta_MDP,state)
            
            nr_arms=numel(state.available_actions);
            
            %The blinkered MDPs are identical across arms. So we only need
            %to solve one of them.
            arm_states = setdiff(meta_MDP.getDownStreamStates(state),state.s);
            
            nr_branches=0; nr_leafs=0;
            for as=1:numel(arm_states)
                nr_choices=sum(any(state.T(arm_states(as),:,:)));
                if nr_choices>1
                    nr_branches=nr_branches+1;
                end
                
                if nr_choices==0
                    nr_leafs=nr_leafs+1;
                end
            end
            
            nr_cells_per_arm = (numel(meta_MDP.getDownStreamStates(state))-1)/nr_arms;
            nr_branches_per_arm=nr_branches/nr_arms;
            nr_leafs_per_branch=nr_leafs/nr_branches;
            
            nr_hallway_cells_per_arm = nr_cells_per_arm - nr_leafs_per_branch*nr_branches_per_arm;
                                
            [T_blinkered,R_blinkered,states_blinkered]=...
                oneArmedMouselabMDP(nr_hallway_cells_per_arm,nr_branches_per_arm,...
                nr_leafs_per_branch,meta_MDP.mean_payoff,meta_MDP.std_payoff,...
                meta_MDP.cost_per_click);

            horizon=nr_cells_per_arm+1;
            gamma=1;            
            [V_blinkered, pi_blinkered, ~] = mdp_finite_horizon(T_blinkered, R_blinkered, gamma, horizon);
            
            meta_MDP.Q_blinkered = getQFromV(V_blinkered(:,1),T_blinkered,R_blinkered,gamma);
            meta_MDP.states_blinkered=states_blinkered;
        end
        
        function PR=featureBasedPR(meta_MDP,state,computation)
           
            Q_c=meta_MDP.predictQ(state,computation);
            
            available_actions=meta_MDP.getActions(state);
            
            Qs=zeros(numel(available_actions),1);
            for a=1:numel(available_actions)
                Qs(a)=meta_MDP.predictQ(state,available_actions(a));
            end
            
            PR=Q_c-max(Qs);
        end
        
        function Q_hat=predictQ(meta_MDP,state,computation)
            weights=meta_MDP.PR_feature_weights;
                        
            Q_hat=weights.VPI*meta_MDP.computeVPI(state,computation)+...
               weights.VOC1*meta_MDP.myopicVOC(state,computation)+...
               weights.ER*max(state.mu_Q(state.s,:));               
        end
    end
    
end