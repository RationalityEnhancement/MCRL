DEBUG = true

if DEBUG
  console.log """
  X X X X X X X X X X X X X X X X X
   X X X X X DEBUG  MODE X X X X X
  X X X X X X X X X X X X X X X X X
  """
  condition = 7
  workerId = ['debugFRED']

  
else
  console.log """
  # =============================== #
  # ========= NORMAL MODE ========= #
  # =============================== #
  """
if mode is "{{ mode }}"
  # Viewing experiment not through the PsiTurk server
  DEMO = true
  condition = 6
  workerId = ['debugFRED']
  # counterbalance = 0


_.mapObject = mapObject
_.compose = _.flowRight
#SHOW_PARTICIPANT_DATA = '0/108'
SHOW_PARTICIPANT_DATA = false
###
All Mouselab-MDP trials will be demonstration trials
with data for the given participant. The coding is
CONDITION/PID and you can find the available codes
in exp1/static/json/data/1B.0/traces
###

experiment_nr = 0.994  #0.992  #1.5 #0.991  # pilot experiment with low-cost and i.i.d. rewards

switch experiment_nr
  when 0 then IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['none','featureBased','fullObservation'], messageTypes: ['full','none'],infoCosts: [0.01,2.80]}    
  when 0.6 then IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['featureBased','none'], messageTypes: ['full','none'],infoCosts: [0.01,1.00,2.50]}
  when 0.9 then IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['featureBased','none','object_level'], messageTypes: ['full'],infoCosts: [0.01,1.00,2.50]}    
  when 0.95 then IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['none'], messageTypes: ['none'],infoCosts: [1.0001]}
  when 0.96 then IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['featureBased'], messageTypes: ['full'],infoCosts: [1.0001]}
  when 0.97 then IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['none'], messageTypes: ['none'],infoCosts: [1.00]} 
  when 0.98 then IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['none'], messageTypes: ['none'],infoCosts: [1.00]} 
  when 0.99 then IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['none'], messageTypes: ['none'],infoCosts: [1.00]} 
  when 0.991 then IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['none'], messageTypes: ['none'],infoCosts: [0.25, 1.00, 4.00]} 
  when 0.992 then IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['none'], messageTypes: ['none'],infoCosts: [0.25], time_limits: [true,false]} 
  when 0.993 then IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['none'], messageTypes: ['none'],infoCosts: [0.01, 0.05,1.25,2.50,2.95,3.50,3.95], time_limits: [true]}
  when 0.994 then IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['none'], messageTypes: ['none'],infoCosts: [0.01, 0.05,0.10], time_limits: [true]} 
  when 1 then IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['none','featureBased','objectLevel'], messageTypes: ['full','none'],infoCosts: [0.01,1.00,1.0001],time_limits:[true]}
  when 1.5 then IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['none','featureBased','objectLevel'], messageTypes: ['full','none'],infoCosts: [0.25,1.00,4.00],time_limits:[true]}    
  when 1.6 then IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['none','featureBased','objectLevel'], messageTypes: ['full','none'],infoCosts: [0.25,1.00,4.00],time_limits:[true]}    
  when 2 then   IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['none','featureBased'], messageTypes: ['full','simple'],infoCosts: [1.00],time_limits:[true]}
  when 3 then IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['none','featureBased','demonstration'], messageTypes: ['full'],infoCosts: [1.00],time_limits:[true]}    
  when 4 then IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['none'], messageTypes: ['full'],infoCosts: [1.00],time_limits:[true]}
  #when 4 then IVs = {IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['none','featureBased'], messageTypes: ['full','simple'],infoCosts: [1.00]}}
  else console.log "Invalid experiment_nr!" 
        
nrDelays = IVs.PRTypes.length    
nrMessages = IVs.messageTypes.length
nrInfoCosts = IVs.infoCosts.length


nrConditions = switch experiment_nr
    when 0 then 6
    when 0.6 then 6
    when 0.9 then 6
    when 0.95 then 1
    when 0.96 then 1
    when 0.992 then 2
    when 0.993 then 4
    when 1 then 3 * 3
    when 1.5 then 3*3
    when 1.6 then 3*3
    else nrDelays * nrMessages * nrInfoCosts

conditions = {'PRType':[], 'messageType':[], 'infoCost': [], 'frequencyOfFB': [],'time_limits': []}

for PRType in IVs.PRTypes
    if experiment_nr < 2 or experiment_nr == 3
        if PRType is 'none'
            messageTypes = ['none']
        else
            messageTypes = ['full']
    else
        messageTypes = IVs.messageTypes
                
    for message in messageTypes            
        for infoCost in IVs.infoCosts      
            for frequency in IVs.frequencyOfFB
                for time_limit in IVs.time_limits
                    conditions.PRType.push(PRType)
                    conditions.messageType.push(message)
                    conditions.infoCost.push(infoCost)
                    conditions.frequencyOfFB.push(frequency)
                    conditions.time_limits.push(time_limit)
          
PARAMS =
  PR_type: conditions.PRType[condition % nrConditions]
  feedback: conditions.PRType[condition % nrConditions] != "none" and conditions.PRType[condition % nrConditions] != "demonstration"
  info_cost: conditions.infoCost[condition % nrConditions]
  message:  conditions.messageType[condition % nrConditions]
  frequencyOfFB: conditions.frequencyOfFB[condition% nrConditions]
  condition: condition
  bonus_rate: 0.01
  delay_hours: 24
  delay_window: 4
  time_limit: conditions.time_limits[condition % nrConditions]    

PARAMS.q_weights = loadJson('static/json/q_weights.json')[PARAMS.info_cost.toFixed(2)]


if experiment_nr is 4
  STAGE1 = true
  STAGE2 = false
  RETURN_TIME = new Date (getTime() + 1000 * 60 * 60 * PARAMS.delay_hours)

if DEBUG
  PARAMS.message = 'full'
  PARAMS.PR_type = 'objectLevel'#'featureBased'
  PARAMS.info_cost = 1.00

# console.log 'PARAMS', PARAMS
COST_LEVEL =
  switch PARAMS.info_cost
    when 0.01 then 'low'
    when 0.05 then 'low'
    when 0.10 then 'low'
    when 0.25 then 'low'
    when 1.00 then 'med'
    when 1.25 then 'med'
    when 2.50 then 'high'
    when 2.95 then 'high'
    when 3.50 then 'high'
    when 3.95 then 'high'
    when 4.00 then 'high'
    when 1.0001 then 'high'
    else throw new Error('bad info_cost')

if PARAMS.time_limit        
    MIN_TIME = 
        switch COST_LEVEL
            when 'low' then 38
            when 'med' then 49
            when 'high' then 39
else
    MIN_TIME = 1