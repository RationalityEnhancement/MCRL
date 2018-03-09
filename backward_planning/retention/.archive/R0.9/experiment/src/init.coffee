DEBUG = true
TALK = no
SHOW_PARTICIPANT = false
STAGE = 1

STAGE1 = STAGE == 1
STAGE2 = STAGE == 2

if DEBUG
  console.log """
  X X X X X X X X X X X X X X X X X
   X X X X X DEBUG  MODE X X X X X
  X X X X X X X X X X X X X X X X X
  """
  condition = 1
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
  condition = 1
  workerId = ['debugFRED']
  # counterbalance = 0
 
CONDITION = parseInt condition
    
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

with_feedback = CONDITION > 0    

PARAMS =
  feedback: condition > 0
  inspectCost: 1
  message:  conditions.messageType[condition % nrConditions]
  frequencyOfFB: conditions.frequencyOfFB[condition% nrConditions]
  condition: condition
  bonusRate: .002
  delay_hours: 24
  delay_window: 4
  time_limit: conditions.time_limits[condition % nrConditions]  
  branching: '312'
  with_feedback: with_feedback
  condition: CONDITION   
  startTime: Date(Date.now())
  variance: '2_4_24'    


RETURN_TIME = new Date (getTime() + 1000 * 60 * 60 * PARAMS.delay_hours)

MIN_TIME = 7