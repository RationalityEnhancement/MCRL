classdef GaussianMetaMDP < MDP
    
    properties
        min_payoff;
        max_payoff;
        mean_payoff=1;
        std_payoff=2;
        nr_cells;
        nr_possible_payoffs;
        payoff_values;
        p_payoff_values;
        reward_function;
        rewards;
        cost_per_click=0.05; %each click costs 5 cents
        cost_per_operation=0.005; %each planning operation costs 1 cents
        total_nr_clicks;
        remaining_nr_clicks;
        add_pseudorewards;
        pseudoreward_type;
        object_level_MDPs;
        object_level_MDP;
        episode=0;
        action_features=7:16;
        feature_names;
        next_states;
        p_next_states;
        query_state;
        query_computation;
        
    end
    
    methods
        function meta_MDP=GaussianMetaMDP(add_pseudorewards,pseudoreward_type,mean_payoff,std_payoff,object_level_MDPs)
            
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
            
            feature_names={state_feature_names{:}, action_feature_names{:}};
            
            meta_MDP.feature_names=feature_names;
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
        
        function [state,meta_MDP]=sampleS0(meta_MDP)
            
            %sample number of outcomes and number of gambles
            state.observations=NaN(meta_MDP.nr_cells,1);
            state.S=meta_MDP.object_level_MDP.states;
            state.T=meta_MDP.object_level_MDP.T;
            state.A=meta_MDP.object_level_MDP.actions;
            
            state.nr_steps=meta_MDP.object_level_MDP.horizon;
            state.has_plan=false;
            
            state.returns=[];
            state.plan=[];
            state.decision=NaN;
            state.outcome=NaN;

            state.step=1;
            state.nr_steps=3;
            state.s=1;
            
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
                        
                        
                        if isnan(state.observations(next_state.nr))
                            state.mu_Q(node.nr,action)=meta_MDP.mean_payoff+state.mu_V(next_state.nr);
                            state.sigma_Q(node.nr,action)=sqrt(meta_MDP.std_payoff^2+state.sigma_V(next_state.nr)^2);
                        else
                            state.mu_Q(node.nr,action)=state.observations(next_state.nr)+state.mu_V(next_state.nr);
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
            
            if c.is_computation
                if isnan(state.observations(c.state)) %not observed yet
                    
                    state.observations(c.state)=meta_MDP.object_level_MDP.rewards(c.from_state,c.state,c.move);
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
                
                r=meta_MDP.object_level_MDP.rewards(state.s,next_state.s,c.decision)-...
                    meta_MDP.costOfPlanning(state,c);
                
                
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
                end
            else
                PR=0;
            end
                        
        end
        
        function [state,decision]=decide(meta_MDP,state,c)
            %determine a move according to the decision mechanism c
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

            decision=c.decision;
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
            state_features=meta_MDP.extractStateFeatures(state);
            action_features=meta_MDP.extractActionFeatures(state,action);
            
            state_action_features=[state_features;action_features];
            
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
                
                VOC=0;
                %nr_cells_uninspected=sum(isnan(state.observations(:)));
                
                regret_reduction=0;
                
                cost=meta_MDP.costOfPlanning(state,c);
                uncertainty_reduction=0;
            else
                expected_regret=0;
                [regret_reduction,meta_MDP]=expectedRegretReduction(meta_MDP,state,c);
                [VOC,meta_MDP]=meta_MDP.myopicVOC(state,c);
                [uncertainty_reduction,meta_MDP]=meta_MDP.expectedUncertaintyReduction(state,c);
                sigma_R=0;
                p_best_action=0;
                sigma_best_action=0;
                underplanning=0;
                complete_planning=0;
                cost=meta_MDP.cost_per_click;
            end
            
            action_features=[expected_regret;regret_reduction;VOC;...
                uncertainty_reduction; sigma_R;p_best_action;sigma_best_action;...
                underplanning;complete_planning;cost];
            
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
        
        function downstream=getDownStreamStates(meta_MDP,state)
                downstream=[];
                
                states=state.S;
                current_path=states(state.s).path;
                steps=1:numel(current_path);
                
                if isempty(current_path)
                    downstream=1:numel(states);
                else                    
                    for s=1:numel(states)
                        if length(states(s).path)>length(current_path)
                            if all(states(s).path(steps)==current_path)
                                downstream=[downstream; s];
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
                if isnan(state.observations(c.state))
                    location=state.s;
                    
                    [~,a_old]=max(state.mu_Q(location,:));
                    
                    [next_states,p_next_states,meta_MDP]=meta_MDP.predictNextState(state,c);
                    
                    delta_EV=zeros(numel(next_states),1);
                    for s=1:numel(next_states)
                        max_posterior_EV=max(next_states(s).mu_Q(location,:));
                        delta_EV(s)=max_posterior_EV-next_states(s).mu_Q(location,a_old);
                    end
                    
                    VOC=dot(p_next_states,delta_EV)-meta_MDP.cost_per_click;
                else
                    VOC=0;
                end
            elseif action.is_decision
                VOC=0;
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
        
    end
    
end