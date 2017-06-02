function properties=evaluateMouselabMDP(T,R,s0,horizon,myopic_first_move,...
    optimal_planning_horizon)

if not(exist('optimal_planning_horizon','var'))
    optimal_planning_horizon=horizon;
end

nr_locations=size(T,1);
nr_actions=size(T,3);
R_hat=-10^4*ones(nr_locations,nr_locations,nr_actions); %impossible actions will be assigned a reward of -inf
T_tilde=repmat(eye(size(squeeze(T(:,:,1)))),[1,1,size(T,3)]);

available_actions_by_state={[1,2,3,4],[1],[2],[3],[4],[2,4],[1,3],[2,4],[1,3],[],[],[],[],[],[],[],[]};

for from=1:nr_locations
    
    available_actions=available_actions_by_state{from};
    
    if isempty(available_actions)
        T_tilde(from,:,:)=repmat(deltaDistribution(from,1:nr_locations),[1,1,nr_actions]);
        R_hat(from,:,:)=zeros(nr_locations,nr_actions);
    end
    
    for a=1:numel(available_actions)
        
        action=available_actions(a);
        T_tilde(from,:,action)=T(from,:,action);
        to=find(squeeze(T_tilde(from,:,action)));
        
        R_hat(from,to,action)=R(from,to,action);

    end
end


if not(exist('myopic_first_move','var'))
    myopic_first_move=false;
end

%1. compute policies that result from different amounts of planning
%a) full-horizon planning
[V, optimal_policy, ~] = mdp_finite_horizon(T_tilde, R_hat, 1, horizon);
properties.R_max=V(s0,1);

properties.optimal_path=[1];
for t=1:horizon
    state=properties.optimal_path(t);
    next_state=find(squeeze(T_tilde(state,:,optimal_policy(state,t))));
    properties.optimal_path=[properties.optimal_path; next_state];
end

minus_R_hat=-R_hat; minus_R_hat(T(:)==0)=-10^6;
[V_neg, worst_policy, ~] = mdp_finite_horizon(T_tilde, minus_R_hat, 1, horizon);
properties.R_min=-V_neg(s0,1);

%b) myopic policies
for h=1:horizon-1 %planning horizon    
    for t=1:horizon %time step
        remaining_steps=horizon-t+1;
        if remaining_steps<=h
            effective_time_step=t;
        else
            effective_time_step=horizon-h+1;
        end
                
        policies(:,t,h)=optimal_policy(:,effective_time_step);
    end
end
policies(:,:,horizon)=optimal_policy;

%2. evaluate those policies
nr_states=size(T,1);
states=1:nr_states;
terminal_states=10:17;
P0=deltaDistribution(s0,states)';
actions=1:size(T,3);
PRs=zeros(size(R));
mdp_sim=MDPSimulator(states,actions,T,R,P0,terminal_states,PRs,horizon,available_actions_by_state);

agent=PlanExecuter(mdp_sim);
for h=horizon:-1:1
    agent.plan=policies(:,:,h);
    agent.has_plan=true;
    [~,R_total(h+1),episodes{h+1},~]=mdp_sim.simulate(agent,1,1,horizon);
end

%no planning
agent.has_plan=false;
[~,R_total(1),episodes{1},~]=mdp_sim.simulate(agent,1,100,horizon);

horizons=0:horizon;
for h=1:numel(horizons)
    for t=1:horizon
        state=episodes{h}.states{1}(t);
        action=episodes{h}.actions{1}(t);
        optimal_action=optimal_policy(state,t);
        if (action~=optimal_action)
            properties.error_on_move(t,h)=true;
        else
            properties.error_on_move(t,h)=false;
        end
    end
    properties.nr_suboptimal_moves(h)=sum(double(properties.error_on_move(:,h)));
end


properties.R_total=[R_total.mean];
properties.delta_R=properties.R_total-max(properties.R_total);

properties.benefit_of_planning=sum(properties.R_total(optimal_planning_horizon+1)-...
    properties.R_total(1:optimal_planning_horizon))-(properties.R_total(end)-properties.R_total(optimal_planning_horizon+1));

properties.percent_suboptimal_moves=properties.nr_suboptimal_moves/horizon;
properties.first_move_optimal=not(properties.error_on_move(1,:));
properties.myopic_first_move_optimal=properties.first_move_optimal(2);

properties.non_negative=properties.R_min>=0;

%properties.nr_suboptimal_actions=

%short_h=min(3,horizon);
%if any(or(or(properties.percent_suboptimal_moves(1:short_h)==0,...
%        properties.delta_R(1:short_h)==0),...
%        properties.first_move_optimal(1:short_h)))
%    properties.score=0;
%elseif or(properties.R_total(end)<0,all(properties.R_total>0))
%    properties.score=0;
%else
    properties.score=sum(properties.nr_suboptimal_moves(1:optimal_planning_horizon))-...
        properties.nr_suboptimal_moves(optimal_planning_horizon+1)+...
        properties.benefit_of_planning;
%end

if myopic_first_move
    if not(properties.myopic_first_move_optimal)
        properties.score=0;
    else
        properties.score=sum(properties.nr_suboptimal_moves)+properties.benefit_of_planning;        
    end
    
    if any(abs(R(:))>9.99)
        properties.score=0;
    end
        
    max_score=1+5;
    properties.optimality=properties.score/max_score;
else   
    max_score=sum(1:horizon-1)+30;
    properties.optimality=properties.score/max_score;
end

if not(properties.non_negative)
    properties.score=0;
end


end