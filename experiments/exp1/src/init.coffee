DEBUG = no

if DEBUG
  console.log """
  X X X X X X X X X X X X X X X X X
   X X X X X DEBUG  MODE X X X X X
  X X X X X X X X X X X X X X X X X
  """
  console.log 'FOOBAR'
  # condition = 2
  
else
  console.log """
  # =============================== #
  # ========= NORMAL MODE ========= #
  # =============================== #
  """
if mode is "{{ mode }}"
  # Viewing experiment not through the PsiTurk server
  DEMO = true
  condition = 3
  # counterbalance = 0


#SHOW_PARTICIPANT_DATA = '0/108'
SHOW_PARTICIPANT_DATA = false
###
All Mouselab-MDP trials will be demonstration trials
with data for the given participant. The coding is
CONDITION/PID and you can find the available codes
in exp1/static/json/data/1B.0/traces
###

experiment_nr = 1

switch experiment_nr
  when 0 then IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['none','featureBased','fullObservation'], messageTypes: ['full','none'],infoCosts: [0.01,2.80]}    
  when 0.6 then IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['featureBased','none'], messageTypes: ['full','none'],infoCosts: [0.01,1.00,2.50]}
  when 0.9 then IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['featureBased','none','object_level'], messageTypes: ['full'],infoCosts: [0.01,1.00,2.50]}    
  when 1 then IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['none','featureBased','objectLevel'], messageTypes: ['full','none'],infoCosts: [0.01,1.00,2.50]}
  when 2 then   IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['none','featureBased'], messageTypes: ['full','simple'],infoCosts: [1.00]}
  when 3 then IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['none','featureBased','demonstration'], messageTypes: ['full'],infoCosts: [1.00]}        
  when 4 then IVs = IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['featureBased'], messageTypes: ['full'],infoCosts: [1.00]}
  #when 4 then IVs = {IVs = {frequencyOfFB : ['after_each_move'], PRTypes: ['none','featureBased'], messageTypes: ['full','simple'],infoCosts: [1.00]}}
  else console.log "Invalid experiment_nr!" 
        
nrDelays = IVs.PRTypes.length    
nrMessages = IVs.messageTypes.length
nrInfoCosts = IVs.infoCosts.length


nrConditions = switch experiment_nr
    when 0 then 6
    when 0.6 then 6
    when 0.9 then 6
    when 1 then 3 * 3
    else nrDelays * nrMessages * nrInfoCosts

conditions = {'PRType':[], 'messageType':[], 'infoCost': [], 'frequencyOfFB': []}

for PRType in IVs.PRTypes
    if experiment_nr <= 1
        if PRType is 'none'
            messageTypes = ['none']
        else
            messageTypes = ['full']
    else
        messageTypes = IVs.messageTypes
                
    for message in messageTypes            
        for infoCost in IVs.infoCosts      
            for frequency in IVs.frequencyOfFB
                conditions.PRType.push(PRType)
                conditions.messageType.push(message)
                conditions.infoCost.push(infoCost)
                conditions.frequencyOfFB.push(frequency)
        
  

PARAMS =
  PR_type: conditions.PRType[condition % nrConditions]
  feedback: conditions.PRType[condition % nrConditions] != "none" and conditions.PRType[condition % nrConditions] != "demonstration"
  info_cost: conditions.infoCost[condition % nrConditions]
  message:  conditions.messageType[condition % nrConditions]
  frequencyOfFB: conditions.frequencyOfFB[condition% nrConditions]
  condition: condition
  bonus_rate: 0.005
  delay_hours: 18
  delay_window: 8

if experiment_nr is 4
  STAGE1 = true
  STAGE2 = false
  RETURN_TIME = new Date (getTime() + 1000 * 60 * 60 * PARAMS.delay_hours)

# if DEBUG
  # PARAMS.message = 'full'
  # PARAMS.info_cost = 2.50
  # PARAMS.PR_type = 'featureBased'

console.log 'PARAMS', PARAMS
COST_LEVEL =
  switch PARAMS.info_cost
    when 0.01 then 'low'
    when 1.00 then 'med'
    when 2.50 then 'high'
    else throw new Error('bad info_cost')