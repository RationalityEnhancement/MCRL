// Warn if overriding existing method
if(Array.prototype.equals)
    console.warn("Overriding existing Array.prototype.equals. Possible causes: New API defines the method, there's a framework conflict or you've got double inclusions in your code.");
// attach the .equals method to Array's prototype to call it on any array
Array.prototype.equals = function (array) {
    // if the other array is a falsy value, return
    if (!array)
        return false;

    // compare lengths - can save a lot of time 
    if (this.length != array.length)
        return false;

    for (var i = 0, l=this.length; i < l; i++) {
        // Check if we have nested arrays
        if (this[i] instanceof Array && array[i] instanceof Array) {
            // recurse into the nested arrays
            if (!this[i].equals(array[i]))
                return false;       
        }           
        else if (this[i] != array[i]) { 
            // Warning - two different object instances will never be equal: {x:20} != {x:20}
            return false;   
        }           
    }       
    return true;
}

// Hide method from for-in loops
Object.defineProperty(Array.prototype, "equals", {enumerable: false});
Object.size = function(obj) {
    var size = 0, key;
    for (key in obj) {
        if (obj.hasOwnProperty(key)) size++;
    }
    return size;
};

Array.prototype.diff = function(a) {
    return this.filter(function(i) {return a.indexOf(i) < 0;});
};
// Hide method from for-in loops
Object.defineProperty(Array.prototype, "diff", {enumerable: false});




meta_MDP=metaMDP()


PRs = new Array()
action_was_click = new Array()
action_was_move = new Array()

moves   = new Array()
clicks  = new Array()

object_level_PRs = loadObjectLevelPRs()

/*
conditions: 1 = optimal delays, optimal message; 0 = fixed delays, no message;
            2 = optimal delays, simple message; 3 = fixed delays, optimal message
*/

conditions_with_delays=[1,2,4,5,6]
conditions_with_metalevel_PR=[1,2,6]
conditions_with_objectlevel_PR=[4,5]

function metaMDP(){

    trials=getTrials()    
    clicks=[]
    
    var meta_MDP={
        cost_per_click: [],
        cost_per_planning_step: 0.01,
        mean_payoff: 4.5,
        std_payoff: 10.6,        
        object_level_MDPs: trials,
        object_level_MDP: [],
        locations: [],
        state: [],
        previous_state: [],
        locations_by_step: [],
        locations_by_path: [],
        action_nrs: {right: 1, up: 2, left: 3, down: 4},
        delay_per_point: [],        
        init: function(problem_nr){
            this.object_level_MDP=this.object_level_MDPs[problem_nr]
            this.locations=getLocations(problem_nr)            
            
            subject_value_of_1h=20; //50 dollars worth of subjective utility per hour
            nr_trials = Object.size(this.object_level_MDPs)
            sec_per_h=3600
            this.delay_per_point = 0.05/(subject_value_of_1h*nr_trials)*sec_per_h;
            
            var state={
                mu_Q: new Array(Object.size(this.locations)),
                sigma_Q: new Array(Object.size(this.locations)),
                mu_V: new Array(Object.size(this.locations)),
                sigma_V: new Array(Object.size(this.locations)),
                s: 1,
                nr_steps: 3,
                step: 1,
                observations: getObservations(clicks,this.locations),
                moves: new Array(Object.size(this.locations))
            }
            
            this.locations_by_step= new Array(4)
            this.locations_by_path= new Array()
            for (s=0; s<this.locations_by_step.length; s++){
                this.locations_by_step[s]=new Array()
            }
            for (l in this.locations){
                locus=this.locations[l]
                this.locations_by_step[locus.path.length].push(locus)                
                this.locations_by_path[locus.path.toString()]=locus
            }
            
            this.state=updateBelief(this,state,_.range(1,state.observations.length+1))
        },
        rational_moves: function(belief_state){
            
            var current_location = meta_MDP.locations[meta_MDP.state.s]            
            var action_index=argmax(belief_state.mu_Q[belief_state.s-1])
            
            var rational_moves = new Array()
            for (var i=0; i<action_index.length; i++){
                rational_moves.push(belief_state.moves[belief_state.s-1][action_index[i]])
            }
            return rational_moves
        }
    }

    return meta_MDP;
}


function getPR(state,action_sequence){
    //getPR returns the sum of the pseudo-rewards taking a sequence of actions in a given state. The result should be zero if all actions are optimal and negative otherwise. 
    //state: state struct
    //action_sequence: array of action structs
    
    meta_MDP.cost_per_click=PARAMS.info_cost
    
    var PRs = new Array()
    
    var current_state = deepCopy(state)
    
    for (var i in action_sequence){
        
        Q_a=predictQValue(current_state,action_sequence[i],current_state)
        
        var available_actions=getActions(state)
        
        var Q_values = new Array()
        for (var a in available_actions)
        {
            Q_values.push(predictQValue(current_state,available_actions[a],current_state)) 
        }
        var V_s = _.max(Q_values)
        
        var PR = Q_a-V_s
        //PR=0 if there is only one possible move and it was taken
        if (action_sequence[i].is_move & current_state.mu_Q[current_state.s-1].length==1){
            PR = 0;    
        }
        //if you chose the option with the higher none value in the last step, then PR=0
        var is_last_step=current_state.step==current_state.nr_steps
        
        if(action.is_move & is_last_step){        
            var children=getDownStreamStates(current_state)
            var observed_all_children = true
            var child_rewards= new Array()
            for (c in children){
                var child_value = state.observations[children[c]-1]
                child_rewards.push(child_value)
                if (isNaN(child_value) | child_value == null){
                    var observed_everything=false
                    break
                }
            }        
            var chose_well = state.observations[action_sequence[i].move.next_state-1] == _.max(child_rewards)

            if (is_last_step & observed_all_children & chose_well){
                PR=0
            }
        }
        PRs.push(PR)
        
        current_state = getNextState(current_state,action_sequence[i])
    }
    
    return sum(PRs)
    
}

/*
function getPROld(state,actions){
    meta_MDP.cost_per_click = PARAMS.info_cost
  
    var next_state=getNextState(state,actions,true)
    var environment_model=getNextState(state,actions.slice(0,-1),true) //information state after having thought but before having taken action
      
    var V_new=valueFunction(next_state,environment_model)
    var V_old=valueFunction(state,environment_model)
    
    var reward=0
    
    for (a in actions){
        if (actions[a].is_move){
            //if (isNaN(environment_model.observations[actions[a].move.next_state-1]) || environment_model.observations[actions[a].move.next_state-1]==null){
            if (isNaN(environment_model.observations[actions[a].move.next_state-1]) || environment_model.observations[actions[a].move.next_state-1]==null){
                var ER=meta_MDP.mean_payoff
            }
            else{
                var ER=environment_model.observations[actions[a].move.next_state-1];
            }
            
            reward+=ER-costOfPlanning(state,actions[a].planning_horizon)
        }
        else{
            reward-=meta_MDP.cost_per_click
        }
    }
    
    var PR=V_new-V_old+reward

    return PR
    //To improve the quality of the feedback, we could use E[V(S_{t+1})] instead of V(s_{t+1}).
    //This would take longer to compute and would also be more effort to implement.
}
*/


function computeDelay(initial_state,actions){   
    //returns the delay in seconds corresponding to the PR for starting in initial_state and taking the actions in the array actions
    
    // if (_.contains(conditions_with_metalevel_PR,condition)){
    //     total_PR=getPR(initial_state,actions)
    // }
    
    // if (_.contains(conditions_with_objectlevel_PR,condition)){
    //     PRs=getObjectLevelPR(meta_MDP.object_level_MDP.trialID, initial_state, actions)
    //     total_PR=sum(PRs)
    // }
    
    total_PR=getPR(initial_state,actions)
    delay=-meta_MDP.delay_per_point*total_PR
    console.log('delay', total_PR, delay)
    
    // delay = delay*10
    
    return delay
    
}

function getObjectLevelPR(problem_nr,initial_state,actions){
    all_PRs=object_level_PRs[problem_nr]

    PRs= new Array()

    from=meta_MDP.state.s
    
    nr_actions=actions.length
    
    for (var a=0; a<nr_actions; a++){
        
        if (actions[a].is_move){        
            from_location=meta_MDP.locations[from]        
            to=actions[a].move.next_state            
            
            PRs.push(all_PRs[meta_MDP.state.step-1][from-1][to-1])

            from= to;
        }
    }
    
    return PRs        
}

function registerMove(direction){    
    //returns the delay and type of feedback message
    //adds the move to the array moves
    //updates the state of meta_MDP
    //direction is a string ("right","up","left", or "down")
    
    var last_move=action_was_move.lastIndexOf(true)
    
    action_was_click.push(false)
    action_was_move.push(true)

    var current_location = meta_MDP.locations[meta_MDP.state.s]
    var available_moves=getMoves(current_location)
    
    var action_nr=meta_MDP.action_nrs[direction]
    
    
    for (a in available_moves){
        
        if (available_moves[a].move.action_nr==action_nr){
            var move = {
                is_move: true,
                is_click: false,
                cell: [],
                move: {
                    next_state: available_moves[a].move.next_state,
                    reward: available_moves[a].move.reward,
                    direction: direction,
                    action_nr: action_nr
                },
                planning_horizon: meta_MDP.state.nr_steps-meta_MDP.state.step+1
            }

            moves.push(move)
        }
    }
    
    var delay=computeDelay(meta_MDP.state,clicks.concat([move]))
    
    var last_move = moves.slice(-1).pop()
    
    var updated_belief=deepCopy(meta_MDP.state)
    updated_belief.observations=getObservations(clicks,meta_MDP.locations)
    updated_belief=updateBelief(meta_MDP,updated_belief,_.range(1,updated_belief.observations.length+1))
    var information_used_correctly= _.contains(meta_MDP.rational_moves(updated_belief), last_move.move.direction)

    //check if all of the successor states have been inspected
    var downstream=getDownStreamStates(meta_MDP.state)
        
    var planned_too_little=false
    
    var available_moves = getMoves(meta_MDP.locations[updated_belief.s])
    if (Object.size(available_moves)>1){ //it is impossible to plan too much if there is no choice    
        for (var u in downstream){
            if (isNaN(updated_belief.observations[downstream[u]-1]) || updated_belief.observations[downstream[u]-1] == null){
                planned_too_little=true
            }
        }
        var inevitable=[]
    }
    else{
        var inevitable= [available_moves[0].move.next_state];
    }
    
    
    if (available_moves.length==1){
        
    }
    var relevant=setDiff(downstream,inevitable)
    var planned_too_much=false
    for (c in clicks){
        
        if (!_.contains(relevant, clicks[c].cell)){
            planned_too_much=true
        }
    }
    
    meta_MDP.state=getNextState(meta_MDP.state,clicks.concat([move]),true)    
    
    clicks=[]

    
    return {delay: delay,
            planned_too_little: planned_too_little,
            planned_too_much: planned_too_much,
            information_used_correctly: information_used_correctly
           }
}

function registerClick(cell_nr){
    action_was_click.push(true)
    action_was_move.push(false)
    
    click = {
        is_move: false,
        is_click: true,
        cell: cell_nr,
        move: []
    }        
    
    clicks.push(click)
}

function getNextState(state,actions,update_belief){
    //getNextState(s,actions,update_belief) returns the state that results from taking a series of actions in state s without changing state s
    //If update_belief is false, then the location will be udpated but mu_Q and sigma_Q won't be updated.
    
    
    if (update_belief === undefined){
        var update_belief=true;
    }
    
    var next_state=deepCopy(state)
    
    if (!(actions.constructor === Array) ){
        temp=clone(actions)
        actions = new Array()
        actions.push(temp)
    }
    
    at_least_one_click=false
    var observed_outcomes = new Array()
    
    for (a in actions){

        action= actions[a]
        
        if (action.is_click){        
            next_state.observations[action.cell-1]=meta_MDP.locations[action.cell].reward
            
            if (!isNumber(next_state.observations[action.cell-1])){
                alert('something went wrong: action.cell='+action.cell)
            }
            
            at_least_one_click=true
            observed_outcomes.push(action.cell)
        }
    
        if (action.is_move){            
            next_state.s=action.move.next_state
            next_state.observations[next_state.s-1]=meta_MDP.locations[next_state.s].reward
            observed_outcomes.push(next_state.s)
            next_state.step++
        }
    }
     
    if (update_belief){
        next_state=updateBelief(meta_MDP,next_state,observed_outcomes)
    }


    
    return next_state
}

function getActions(state){
    
    var actions=new Array()
    
    var current_location=meta_MDP.locations[state.s]
    
    actions=getMoves(current_location).concat(getClicks(state))
        
    return actions
}

function getMoves(current_location){
    //returns the moves available in the current location
    
    //moves
    var moves=new Array()
    
    var available_actions=current_location.actions
    for (move in available_actions){
        
        move={
            is_move: true,
            is_click: false,
            cell: [],
            move: {
                next_state: available_actions[move][0],
                reward: available_actions[move][1],
                direction: move,
                action_nr: meta_MDP.action_nrs[move]
            }
        }
        
        moves.push(move)
    }
    
    return moves
}

function getClicks(state){
    
    //clicks
    var clicks = new Array()
    
    for (o in state.observations){
        
        if (isNaN(state.observations[o]) || state.observations[o]==null){
            
            click={
                is_move: false,
                is_click: true,
                cell: parseInt(o),
                move: []
            }
            
            clicks.push(click)
        }
    }
    
    return clicks
}

function getLocations(problem_nr){
    
    var locations = meta_MDP.object_level_MDPs[problem_nr].graph
    
    for (l in locations){
        
        locations[l].nr=parseInt(l);
        
        if (isEmpty(locations[l].path)){
            locations[l].path=[]
        }
        else{        
            if (isScalar(locations[l].path)){
                locations[l].path=[locations[l].path]
            }
        
            if (!isScalar(locations[l].path[0])){
                locations[l].path=[].concat.apply([],locations[l].path)
            }
        }
    }
    return locations
}

function getTrials(){
    var experiment = loadJson("static/json/condition_1.json");
    var trials=experiment.trials;
    
    //TODO: Move this information into the JSON file. Otherwise this code won't generalize to other layouts.
    for (t in trials){
        trials[t].leafs=_.range(10,18)
        trials[t].hallway_states=_.range(2,10)
        trials[t].siblings_by_state=[[],[],[3,4,5],[2,4,5],[2,3,5],[2,3,4],[],[],[],[],[11],[10],[13],[12],[15],[14],[17],[16]]
    }
    
    return trials
}

function loadObjectLevelPRs(){
    PR_json = loadJson("static/json/ObjectLevelPRs.json")
    object_level_PRs=PR_json
    /*
    object_level_PRs=new Array()
    for (t=0; t<PR_json.length; t++){    
        var A = reshapeArray(PR_json[t]._ArrayData_,PR_json[t]._ArraySize_)
        object_level_PRs.push(transpose(A))        
    }
    */
    return object_level_PRs
}


function getObservations(clicks,locations){
    var observations=new Array(Object.size(locations))
    
    for (var o=0; o<observations.length; o++){
        observations[o]=NaN;
    }
    
    
    for (c in clicks){
            observations[clicks[c].cell-1] = locations[clicks[c].cell].reward           
    }
    
    return observations
}

function updateBelief(meta_MDP,state,new_observations){
    /* 
        computes state.mu_Q, state.sigma_Q, state.mu_V, and
        state.sigma_V from meta_MDP.observations and meta_MDP
        mu_Q(s,a): expected return of starting in object-level state s and performing object-level action a according to meta_MDP.observations and meta_MDP.p_payoffs 
        mu_V(s): expected return of starting in object-level state s and following the optimal object-level policy according to meta_MDP.observations and meta_MDP.p_payoffs. The expectation is taken with respect to the probability distribution encoded by the meta-level state and the ?optimal policy? maximizes the reward expected according to the probability distribution encoded by the meta-level state
        sigma_Q(s,a), sigma_V(s): uncertainty around the expectations mu_Q(s,a) and mu_V(s).
        
        The value of new_observations should not affect the result, it is only used to speed up the computation.
        
        This function should be equivalent to the updateBelief method of MouselabMDPMetaMDPNIPS.m
    */
    
    //0. Determine which beliefs have to be updated
    var needs_updating=getUpStreamStates(new_observations);


    //1. Set value of starting from the leave states to zero
    var leaf_nodes=meta_MDP.locations_by_step[state.nr_steps];
    for (var l=0; l<leaf_nodes.length; l++){
        state.mu_V[leaf_nodes[l].nr-1]=0;
        state.sigma_V[leaf_nodes[l].nr-1]=0;
    }
        
    
    //2. Propagage the update backwards towards the initial state
    nr_steps=meta_MDP.locations_by_step.length
    
    for (var step=nr_steps-2; step>=0; step--){

        var nodes=meta_MDP.locations_by_step[step]
        // a) Update belief about state-action values
        for (var n=0; n<nodes.length; n++){
            var node=nodes[n];

            if (_.contains(needs_updating,node.nr)){
                state.mu_Q[node.nr-1]=new Array()
                state.sigma_Q[node.nr-1]=new Array()
                state.moves[node.nr-1] = new Array()
                
                available_actions=node.actions
                
                for (var a in available_actions){
                    var action=node.actions[a];
                    
                    var action_nr=meta_MDP.action_nrs[a];
                    
                    var next_state=meta_MDP.locations_by_path[node.path.concat(action_nr).toString()];

                    if (isNaN(state.observations[next_state.nr-1]) || state.observations[next_state.nr-1]==null){
                        state.mu_Q[node.nr-1][action_nr-1]=meta_MDP.mean_payoff+state.mu_V[next_state.nr-1];
                        //state.mu_Q[node.nr-1].push(meta_MDP.mean_payoff+state.mu_V[next_state.nr-1]);
                        state.sigma_Q[node.nr-1][action_nr-1]=Math.sqrt(Math.pow(meta_MDP.std_payoff,2)+Math.pow(state.sigma_V[next_state.nr-1],2));
                        //state.sigma_Q[node.nr-1].push(Math.sqrt(Math.pow(meta_MDP.std_payoff,2)+Math.pow(state.sigma_V[next_state.nr-1],2)));
                    }
                    else{
                        state.mu_Q[node.nr-1][action_nr-1]=state.observations[next_state.nr-1]+state.mu_V[next_state.nr-1]
                        //state.mu_Q[node.nr-1].push(state.observations[next_state.nr-1]+state.mu_V[next_state.nr-1])
                        state.sigma_Q[node.nr-1][action_nr-1]=state.sigma_V[next_state.nr-1]                        
                        //state.sigma_Q[node.nr-1].push(state.sigma_V[next_state.nr-1])                        
                    }
                    state.moves[node.nr-1].push(a)
                }

                state.mu_Q[node.nr-1]=state.mu_Q[node.nr-1].filter(function(val) { return val !== undefined; })
                state.sigma_Q[node.nr-1]=state.sigma_Q[node.nr-1].filter(function(val) { return val !== undefined;})                
                //b) Update belief about state value V
                var EV_and_sigma=EVOfMaxOfGaussians(state.mu_Q[node.nr-1],state.sigma_Q[node.nr-1]);
                state.mu_V[node.nr-1]=EV_and_sigma[0];
                state.sigma_V[node.nr-1]=EV_and_sigma[1];

            }
        }

    }           
    
    return state       
}

function valueFunction(state,environment_model){        
    //returns the approximate value function V_PR(state | environment model)  specified by PARAMS.PR_type. state is the argument of the value function and the value is computed with respect to the information in environment_model.
    
    switch(PARAMS.PR_type){
        case 1: //PRs based on the full-observation policy
            var current_location=state.s;
            var planning_horizon=state.nr_steps-state.step+1;
            var planning_cost=0;
            var step=state.step-1;
    
            for (var ph=planning_horizon; ph>0; ph--){
                var start_location=meta_MDP.locations_by_step[step++][0]
                var start_state = {
                    s: start_location.nr,
                    step: step,
                    nr_steps: state.nr_steps,
                    observations: new Array(Object.size(meta_MDP.locations)),
                    mu_Q: new Array(Object.size(meta_MDP.locations)),
                    sigma_Q: new Array(Object.size(meta_MDP.locations)),
                    mu_V: new Array(Object.size(meta_MDP.locations)),
                    sigma_V: new Array(Object.size(meta_MDP.locations)),
                    moves: new Array(Object.size(meta_MDP.locations))
                }
                planning_cost+=costOfPlanning(start_state,ph);
            }

            var downstream=getDownStreamStates(state);

            var to_be_observed=0
            for (var i in state.observations){
                if (_.contains(downstream,parseInt(i))){
                    if (isNaN(state.observations[parseInt(i)]) || state.observations[parseInt(i)] == null ){
                        to_be_observed++
                    }
                }
            }

            var information_cost=meta_MDP.cost_per_click*to_be_observed

            var V=environment_model.mu_V[current_location-1]-planning_cost-information_cost;
            
            break;
        case 2: //feature-based PRs
            console.log('feature-based valueFunction')
            Q_hat = new Array()
            
            available_actions=getActions(state)
            for (a in available_actions){
                Q_hat.push(predictQValue(state,available_actions[a],environment_model))
            }
            
            console.log('Q_hat', Q_hat)
            V = _.max(Q_hat)
            break;
    }
    return V
}

function predictQValue(state,computation){
    //predictQValue(s,c) returns the Q-value our feature-based approximation predicts for performing computation c in state s.
    
    feature_weights = null;
    switch(PARAMS.info_cost){
        case 0.01:
            feature_weights = {VPI: 1.2065, VOC1: 2.1510, ER: 1.5298};//{VPI: 1.1261, VOC1: 1.0934, ER: 1.0142};
            break;
        case 1.6:
            feature_weights = {VPI: 0.6118, VOC1: 1.2708, ER: 1.3215}; //{VPI: 0.1852, VOC1: 0.3436, ER: 0.9455} //{VPI: 0.3199, VOC1: 0.3363, ER: 0.9178};//{VPI: 1.0734, VOC1: 0.0309, ER: 0.5921};
            break;
        case 2.8:
            feature_weights = {VPI: 0.6779, VOC1: 0.7060, ER: 1.2655};//{VPI: -0.5920, VOC1: -0.1227, ER: 0.8685};
            break;

    console.log('weights', feature_weights)

    }
    
    meta_MDP.cost_per_click = PARAMS.info_cost
    
    var VPI = computeVPI(state,computation)
    var VOC1 = computeMyopicVOC(state,computation)
    var ER = computeExpectedRewardOfActing(state)
    
    var Q_hat= feature_weights.VPI*VPI+feature_weights.VOC1*VOC1 + feature_weights.ER*ER
    return Q_hat
}

function computeExpectedRewardOfActing(state){
    //returns the sum of the rewards the agent expects to receive for acting without further deliberation according to the belief encoded by the current state.
    var plan = makePlan(state)
    var ER = evaluatePlan(state,plan)
    
    return ER
}

function makePlan(state){
    //makePlan(state) returns an array of action structs. Each action has the highest state.mu_Q(location,:) value for the location in which it is chosen.
    
    var plan = new Array()

    var location_nr=state.s;
    var step=state.step;
    
    var current_state=deepCopy(state);
    while (step<=state.nr_steps){
        var max_Q = _.max(state.mu_Q[location_nr-1])        
        var a=_.indexOf(state.mu_Q[location_nr-1],max_Q)        
        
        var available_actions=getMoves(meta_MDP.locations[current_state.s])
        //var action = available_actions[a]
        
        /*
        var actions=getActions(current_state)
        */
        
        var action_nrs=new Array()
        for (i in available_actions){
            action_nrs.push(available_actions[i].move.action_nr)                          
        }
        action_nrs.sort()        
        action_nr=action_nrs[a]

        for (i in available_actions){
            if (available_actions[i].move.action_nr==action_nr){
                action = available_actions[i]
                plan.push(action)  
            }
        }

        current_state = getNextState(current_state,action,false)
        location_nr=current_state.s
        step=step+1;
    }
    
    return plan
}

function evaluatePlan(state,plan){
    //evaluatePlan(s,plan) determines the reward the agent expects to receive for executing the plan from the current location according to the beliefs encoded by state s. The expected reward is the sum of observed or unobserved rewards along the path. If the reward is unobserved, then it is replaced by the avg. payoff.
    //plan is an array containing action structs, each of which is a valid move.
    
    var ER=0

    var current_state=deepCopy(state)
    for (var a in plan){
        var action = plan[a]
        current_state=getNextState(current_state,action,false)
        
        if (isNaN(state.observations[current_state.s-1]) | state.observations[current_state.s-1]==null){
            ER+=meta_MDP.mean_payoff
        }
        else{
            ER+=state.observations[current_state.s-1]
        }
            
    }
    
    return ER
}

/*
function computeExpectedRewardOfActingOld(state){
    
    if (state.mu_Q[state.s-1] == null){
        return 0
    }
    
    if (state.mu_Q[state.s-1].length==0){
        return 0
    }
    else{
        return _.max(state.mu_Q[state.s-1])
    }
}
*/

function computeMyopicVOC(state,c){
    //Computes the myoptic VOC (VOC1, Equation 4 in the NIPS paper) in a highly efficient manner. VOC1 is -cost(c) if the computation cannot improve the decision. VOC1 is positive if the expected improvement in decision quality from a single computation is higher than the cost of computation and negative else. The output should be identical to the outputs of the method myopicVOC of MouselabMDPMetaMDPNIPS in Matlab.
     
    if (c.is_click)  {
        
        if (state.mu_Q[state.s-1].length==1){
            return -meta_MDP.cost_per_click
        }
        
        if ((isNaN(state.observations[c.cell]) || state.observations[c.cell]==null)  && c.cell>1){

            if (_.contains(getDownStreamStates(state),c.cell)){
                locations=getLocations(0)
                path=locations[c.cell].path;

                var a=path[state.step-1];                        
                mu_prior=state.mu_Q[state.s-1];
                                


                //if hallway state
                if (_.contains(meta_MDP.object_level_MDP.hallway_states,c.cell)){
                    VOC=myopicVOCAdditive(mu_prior,a);
                }

                if (_.contains(meta_MDP.object_level_MDP.leafs,c.cell)){

                    siblings=meta_MDP.object_level_MDP.siblings_by_state[c.cell];                    
                    sibling_rewards= new Array()
                    for (s in siblings){
                        //if (!isNaN(state.observations[siblings[s]])){
                        
                        if (state.observations[siblings[s]]==null){
                            sibling_rewards.push(NaN)
                        }
                        else{
                            sibling_rewards.push(state.observations[siblings[s]])
                        }
                        //}
                    }
                    
                    if (sibling_rewards.some(function(x) {return isNaN(x)})){
                        //if leaf node with unknown sibling(s)
                        VOC=myopicVOCMaxUnknown(mu_prior,a);
                    }
                    else{
                        //if leaf node with known sibling(s)
                        alternative=_.max(sibling_rewards);
                        VOC=myopicVOCMaxKnown(mu_prior,a,alternative);
                    }
                }
            }
            else{
                VOC=0-meta_MDP.cost_per_click;
            }

        }
        else{
            VOC=0-meta_MDP.cost_per_click;
        }
    }
    else{
            VOC=0;
    }
    
    return VOC
}

function myopicVOCAdditive(mu_prior,a){
//This function evaluates the VOC of inspecting a hallway cell
//downstream of the current state.
//mu_prior: prior means of returns of available actions
//sigma_prior: prior uncertainty of returns of available actions
//a: action about which more information is being collected
//This function should behave in the same way as the method myopicVOCAdditive of MouselabMDPMetaMDPNIPS.m.

mu_sorted = mu_prior.slice(0).sort(function(a, b){return b - a})
mu_alpha = mu_sorted[0]
mu_beta = mu_sorted[1]

appears_best = mu_prior[a-1] == mu_alpha

if (appears_best){
    //information is valuable if it reveals that action c is suboptimal

    //To change the decision, the sampled value would have to be less than ub
    ub=mu_beta+meta_MDP.mean_payoff-mu_alpha;                
    VOC=meta_MDP.std_payoff**2*normPDF(ub,meta_MDP.mean_payoff,meta_MDP.std_payoff)- (mu_alpha-mu_beta)*normCDF(ub,meta_MDP.mean_payoff,meta_MDP.std_payoff) - meta_MDP.cost_per_click;
}
else{
    //information is valuable if it reveals that action is optimal
    //To change the decision, the sampled value would have to be larger than lb.
    lb=mu_alpha+meta_MDP.mean_payoff-mu_prior[a-1];                
    VOC=meta_MDP.std_payoff**2*normPDF(lb,meta_MDP.mean_payoff,meta_MDP.std_payoff)- (mu_alpha-mu_prior[a-1])*(1-normCDF(lb,meta_MDP.mean_payoff,meta_MDP.std_payoff))- meta_MDP.cost_per_click;
}
    
return VOC

}

function myopicVOCMaxUnknown(mu_prior,a){    
//This function evaluates the VOC of inspecting a leaf cell
//downstream of the current state where the values of the other leaf(s) is/are known.
//mu_prior: prior means of returns of available actions            
//a: action about which more information is being collected
//known_alternative: maximum of the other known leafs
//This function should behave identically to the method myopicVOCMaxUnknown of MouselabMDPMetaMDPNIPS.m

mu_sorted = mu_prior.slice(0).sort(function(a, b){return b - a})
mu_alpha = mu_sorted[0]
mu_beta = mu_sorted[1]

appears_best = mu_prior[a-1] == mu_alpha

E_max=EVOfMaxOfGaussians([meta_MDP.mean_payoff,meta_MDP.mean_payoff],
[meta_MDP.std_payoff,meta_MDP.std_payoff]);

if (appears_best){
    //information is valuable if it reveals that action c is suboptimal

    lb=meta_MDP.mean_payoff-3*meta_MDP.std_payoff;
    ub=meta_MDP.mean_payoff;
    delta_x=meta_MDP.std_payoff/20.0;
    VOC=integral(lb,ub,delta_x,function(x){return normPDF(x,meta_MDP.mean_payoff,meta_MDP.std_payoff)*_.max([0,  mu_beta - (mu_alpha-E_max+ETruncatedNormal(meta_MDP.mean_payoff,meta_MDP.std_payoff, x,meta_MDP.mean_payoff+5*meta_MDP.std_payoff))])})-meta_MDP.cost_per_click; 
}
else{
    //information is valuable if it reveals that action is optimal                
    lb=meta_MDP.mean_payoff;
    ub=meta_MDP.mean_payoff+3*meta_MDP.std_payoff;
    delta_x=meta_MDP.std_payoff/20.0;
    
    VOC=integral(lb,ub,delta_x,function(x){return normPDF(x,meta_MDP.mean_payoff,meta_MDP.std_payoff)*_.max([0, (mu_prior[a-1]-E_max+ETruncatedNormal(meta_MDP.mean_payoff,meta_MDP.std_payoff,x,meta_MDP.mean_payoff+5*meta_MDP.std_payoff))-mu_alpha])})-meta_MDP.cost_per_click;                                
}

return VOC

}

function myopicVOCMaxKnown(mu_prior,a,known_alternative){
    //computes the myopic VOC of inspecting a leaf when the rewards of all other leafs are known
    //mu_prior: prior means of the actions available in the current state
    //number of object level action about which information would be collected (branch of the leaf)
    //known_alternative: max. of the rewards of the siblings of the leaf node
//This function should behave identically to the method myopicVOCMaxKnown of MouselabMDPMetaMDPNIPS.m
    
    mu_sorted = mu_prior.slice(0).sort(function(a, b){return b - a})
    mu_alpha = mu_sorted[0]
    mu_beta = mu_sorted[1]

    appears_best = mu_prior[a-1] == mu_alpha

    E_max= EVOfMaxOfGaussians([meta_MDP.mean_payoff,known_alternative],[meta_MDP.std_payoff,0]);

    if (appears_best){
        //information is valuable if it reveals that action c is suboptimal

        //The decision can only change if E[max{known_alternative,x}]-k>mu_alpha-mu_beta

        if (E_max-known_alternative<= mu_alpha-mu_beta){
            VOC=0-meta_MDP.cost_per_click;
        }
        else{
            //to change the decision x would have to be less than the known alternative
            ub=known_alternative;                    
            VOC=normCDF(ub,meta_MDP.mean_payoff,meta_MDP.std_payoff) * (mu_beta-(mu_alpha-E_max+known_alternative))-meta_MDP.cost_per_click;                    
        }                                                
    }
    else{
        //information is valuable if it reveals that action is optimal       
        //To change the decision, the sampled value would have to be larger than lb
        lb=mu_alpha-mu_prior[a-1]+E_max;

        VOC=meta_MDP.std_payoff^2*normPDF(lb,meta_MDP.mean_payoff,meta_MDP.std_payoff)-(mu_alpha-mu_prior[a-1]-E_max-meta_MDP.mean_payoff)*(1-normCDF(lb,meta_MDP.mean_payoff,meta_MDP.std_payoff))-meta_MDP.cost_per_click;
    }

    return VOC
}

function computeVPI(state,metalevel_action){
//returns the value of perfect information (Equation 5 in the NIPS paper)
//This function's input-output behavior should agree with the method computeVPI of MouselabMDPMetaMDPNIPS.m

if (metalevel_action.is_move)
    return 0
    
if (state.mu_Q[state.s-1]==null)
    return 0
    
if (metalevel_action.cell==0)
    return 0

if (state.mu_Q[state.s-1].length<=1){
    return 0 //no choice --> no way that new information can change the decision
}
    
//sort mu_Q values in descending order
mu_sorted = state.mu_Q[state.s-1].slice(0).sort(function(a, b){return b - a})
max_val = mu_sorted[0]
if (mu_sorted.length>=2){
    secondbest_val=mu_sorted[1]
}
else{
    secondbest_val=-Infinity
}

locations=getLocations(0);
path_to_cell=locations[metalevel_action.cell].path

cell_step=path_to_cell.length

if (state.step>cell_step){
    return 0
}

corresponding_action= meta_MDP.locations[metalevel_action.cell].path[state.step-1] 
mu_c=state.mu_Q[state.s-1][corresponding_action-1]
sigma_c = state.sigma_Q[state.s-1][corresponding_action-1]

appears_best = mu_c==max_val

if (appears_best){
    //information is valuable if it reveals that action c is suboptimal
    ub=secondbest_val;
    lb=mu_c-3*sigma_c;
    
    VPI = (secondbest_val-mu_c)*(normCDF(ub,mu_c,sigma_c)-normCDF(lb,mu_c,sigma_c))+sigma_c**2*(normPDF(ub,mu_c,sigma_c)-normPDF(lb,mu_c,sigma_c));    
}
else{
    //information is valuable if it reveals that action is optimal
    ub=mu_c+3*sigma_c;
    lb=max_val;
    
    if (ub>lb){
        VPI = (mu_c-max_val)*(normCDF(ub,mu_c,sigma_c)-normCDF(lb,mu_c,sigma_c))-sigma_c**2*(normPDF(ub,mu_c,sigma_c)-normPDF(lb,mu_c,sigma_c));
    }
    else{
        VPI=0;
    }
}

return VPI
}

function costOfPlanning(state,planning_horizon){
    //compute the number and length of all paths from the current 
    var current_location=meta_MDP.locations[state.s];
    var available_actions=getMoves(meta_MDP.locations[state.s])

    var cost_of_planning=0;
    if (planning_horizon>0){
        for (m in moves){
            var move=moves[m];
            var next_state=getNextState(state,move,false);
            var next_planning_horizon=planning_horizon-1;
            
            cost_of_planning+=meta_MDP.cost_per_planning_step+costOfPlanning(next_state,next_planning_horizon);
        }
    }
    
    return cost_of_planning
}

function getDownStreamStates(state){
    //returns all states that can be reached from state
    
    var downstream=[];

    var states=meta_MDP.locations;
    var current_path=states[state.s].path;
    var path_length=current_path.length

    if (current_path.length==0){
        downstream=_.range(1,Object.size(states));
    }
    else{                    
        for (s=1; s<=Object.size(states); s++){
            if (states[s].path.slice(0,path_length).equals(current_path) && states[s].path.length>path_length){
                downstream.push(s);
            }

        }
    }

    return downstream                
}

function getUpStreamStates(observed_states){
    //returns all states from which any of the observed_states can be reached
    
    var upstream=[];

    var states=meta_MDP.locations
    
    for (o=0; o<observed_states.length; o++){
        var current_path=states[observed_states[o]].path;    

        if (current_path.length>0){                    
            for (s=1; s<=Object.size(states); s++){

                var path_length=states[s].path.length

                if (path_length < current_path.length){
                    if (states[s].path.equals(current_path.slice(0,path_length))){
                        upstream.push(s);
                    }
                }
            }

        }
    }

    return upstream
                
}


checkObj = function(obj, keys) {
  var i, k, len;
  if (keys == null) {
    keys = Object.keys(obj);
  }
  for (i = 0, len = keys.length; i < len; i++) {
    k = keys[i];
    if (obj[k] === void 0) {
      console.log('Bad Object: ', obj);
      throw new Error(k + " is undefined");
    }
  }
  return obj;
};

assert = function(val) {
  if (!val) {
    throw new Error('Assertion Error');
  }
  return val;
};

checkWindowSize = function(width, height, display) {
  var maxHeight, win_width;
  win_width = $(window).width();
  maxHeight = $(window).height();
  if ($(window).width() < width || $(window).height() < height) {
    display.hide();
    return $('#window_error').show();
  } else {
    $('#window_error').hide();
    return display.show();
  }
};
	
function isScalar(obj){
    return (/string|number|boolean/).test(typeof obj);
}

function isEmpty(val){
    return (val === undefined || val == null || val.length <= 0) ? true : false;
}

function deepCopy(some_array){
    return JSON.parse(JSON.stringify(some_array))
}

function clone(obj) {
    if (null == obj || "object" != typeof obj) return obj;
    var copy = obj.constructor();
    for (var attr in obj) {
        if (obj.hasOwnProperty(attr)){
            
            if (typeof(obj[attr]) === "object"){
                copy[attr]=clone(obj[attr])
            }
            else{
                copy[attr] = obj[attr];
            }
        }
    }
    return copy;
}

function sum(vector){
    return vector.reduce(add, 0);
}
function add(a, b) {
    return a + b;
}

function isNumber(n) {
  return !isNaN(parseFloat(n)) && isFinite(n);
}

function argmax(arr) {
    if (arr.length === 0) {
        return -1;
    }

    var max = arr[0];
    var maxIndex = [0];

    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = [i];
            max = arr[i];
        }
        else{
            if (arr[i] == max){
                maxIndex.push(i)
            }
        }
    }

    return maxIndex;
}

function test(){
    
    var condition=1
    
    meta_MDP.init(2)    
    available_actions=getActions(meta_MDP.state)
    click_action=available_actions[7]
    move_action=available_actions[1]
    
    s_next=getNextState(meta_MDP.state,click_action)
    
    PR_click=getPR(meta_MDP.state,click_action)
    PR_move=getPR(meta_MDP.state,move_action)
    
    action_nrs=[7,8,9,10,11,1];    
    actions = new Array()
    for (a in action_nrs){
        actions.push(available_actions[action_nrs[a]])
    }
    
    delay=computeDelay(meta_MDP.state,actions)
        
}

function test2(){
    
    var condition=1
    meta_MDP.init(2)    

    registerClick(2)
    registerClick(3)
    registerClick(4)
    registerClick(5)
    registerMove("up")
}

function recomputeDelays(){
    
    var temp_condition=condition
    
    condition=1
    
    var clicks_and_paths=loadJson("static/js/clicks_and_paths2.json")
    for (var t in clicks_and_paths){
        
        meta_MDP.init(clicks_and_paths[t].trialID)
        
        var move_times=JSON.parse(clicks_and_paths[t].actionTimes)
        var click_times=JSON.parse(clicks_and_paths[t].clickTimes)
        var clicks=JSON.parse(clicks_and_paths[t].clicks.replaceAll("'",""))
        var path=JSON.parse(clicks_and_paths[0].path)
        
        var delays= new Array()
        
        if (click_times.length>0){
            var first_click_before_move=0;
                        
            for (m in move_times){
                var move_time=move_times[m]

                var last_click_before_move=_.findLastIndex(click_times,function(x){return x<move_time})
                
                var clicks_before_move = clicks.slice(first_click_before_move,last_click_before_move+1)
                
                for (c in clicks_before_move){
                    registerClick(clicks_before_move[c])
                }
                
                var available_actions=getMoves(meta_MDP.locations[path[parseInt(m)]])
                
                for (var a in available_actions){
                    if (available_actions[a].move.next_state==path[parseInt(m)+1]){
                        var feedback=registerMove(available_actions[a].move.direction)
                        delays.push(feedback.delay)
                    }
                }
                
                first_click_before_move=last_click_before_move+1;

            }
        }
        clicks_and_paths[t].delays=delays;
        console.log("completed trial "+t+" of "+clicks_and_paths.length)
    }
    
    condition=temp_condition
    
    download(JSON.stringify(clicks_and_paths), 'clicks_and_paths.json', 'text/plain');

}

String.prototype.replaceAll = function(search, replacement) {
    var target = this;
    return target.replace(new RegExp(search, 'g'), replacement);
};

function download(text, name, type) {
    var a = document.createElement("a");
    var file = new Blob([text], {type: type});
    a.href = URL.createObjectURL(file);
    a.download = name;
    a.click();
}

function setDiff(A,B){
    // return A.filter(x => B.indexOf(x) < 0 );
    return A.filter(function(x) {return B.indexOf(x) < 0})
}

function transpose(A){
  return newArray = A[0].map(function(col, i) { 
  return A.map(function(row) { 
    return row[i] 
  })
});
}

function reshapeArray(input_list,size_vector){
    var nr_dimensions=size_vector.length;
    
    if (nr_dimensions==1){
        return input_list;
    }
    //else execute the code below
    
    var N=input_list.length;
    
    //Make a copy of the list
    temp=input_list.slice(0)
    
    var A=new Array()
    
    var nr_elements=N/size_vector[nr_dimensions-1];
    for (e=0; e<size_vector[nr_dimensions-1]; e++){
        var entries = temp.slice(e*nr_elements,(e+1)*nr_elements)
        A.push(reshapeArray(entries,size_vector.slice(0,nr_dimensions-1)))
    }               
    
    return transpose(A)
}