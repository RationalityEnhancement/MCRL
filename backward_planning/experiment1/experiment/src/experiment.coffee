# coffeelint: disable=max_line_length, indentation

DEBUG = no
TALK = no
SHOW_PARTICIPANT = false

if DEBUG
  console.log """
  X X X X X X X X X X X X X X X X X
   X X X X X DEBUG  MODE X X X X X
  X X X X X X X X X X X X X X X X X
  """
  CONDITION = 1

else
  console.log """
  # =============================== #
  # ========= NORMAL MODE ========= #
  # =============================== #
  """
  console.log '16/01/18 12:38:03 PM'
  CONDITION = parseInt condition

if mode is "{{ mode }}"
  DEMO = true
  CONDITION = 1

with_feedback = CONDITION > 0    

BLOCKS = undefined
PARAMS = undefined
TRIALS = undefined
DEMO_TRIALS = undefined
STRUCTURE = undefined
N_TRIAL = undefined
SCORE = 0
calculateBonus = undefined
getTrials = undefined

psiturk = new PsiTurk uniqueId, adServerLoc, mode

psiturk.recordUnstructuredData 'condition', CONDITION   
psiturk.recordUnstructuredData 'with_feedback', with_feedback

saveData = ->
  new Promise (resolve, reject) ->
    timeout = delay 10000, ->
      reject('timeout')

    psiturk.saveData
      error: ->
        clearTimeout timeout
        console.log 'Error saving data!'
        reject('error')
      success: ->
        clearTimeout timeout
        console.log 'Data saved to psiturk server.'
        resolve()


$(window).resize -> checkWindowSize 800, 600, $('#jspsych-target')
$(window).resize()
$(window).on 'load', ->
  # Load data and test connection to server.
  slowLoad = -> $('slow-load')?.show()
  loadTimeout = delay 12000, slowLoad

  psiturk.preloadImages [
    'static/images/spider.png'
  ]


  delay 300, ->
    console.log 'Loading data'
        
    PARAMS =
      inspectCost: 1
      startTime: Date(Date.now())
      bonusRate: .002
      # variance: ['2_4_24', '24_4_2'][CONDITION]
      branching: '312'
      with_feedback: with_feedback
      condition: CONDITION      

    psiturk.recordUnstructuredData 'params', PARAMS

    if PARAMS.variance
      id = "#{PARAMS.branching}_#{PARAMS.variance}"
    else
      id = "#{PARAMS.branching}"
    STRUCTURE = loadJson "static/json/structure/312.json"
    TRIALS = loadJson "static/json/mcrl_trials/increasing.json"
    console.log "loaded #{TRIALS?.length} trials"

    getTrials = do ->
      t = _.shuffle TRIALS
      idx = 0
      return (n) ->
        idx += n
        t.slice(idx-n, idx)

    if DEBUG or TALK
      createStartButton()
      clearTimeout loadTimeout
    else
      console.log 'Testing saveData'
      if DEMO
        clearTimeout loadTimeout
        delay 500, createStartButton
      else
        saveData().then(->
          clearTimeout loadTimeout
          delay 500, createStartButton
        ).catch(->
          clearTimeout loadTimeout
          $('#data-error').show()
        )

createStartButton = ->
  if DEBUG or TALK
    initializeExperiment()
    return
  if DEMO
    $('#jspsych-target').append """
      <div class='alert alert-info'>
        <h3>Demo mode</h3>

        To go through the task as if you were a participant,
        click <b>Begin</b> above.<br>
        To view replays of the participants
        in our study, click <b>View Replays</b> below.
      </div>
      <div class='center'>
        <button class='btn btn-primary btn-lg centered' id="view-replays">View Replays</button>
      </div>
    """
    $('#view-replays').click ->
      SHOW_PARTICIPANT = true
      DEMO_TRIALS = _.shuffle loadJson "static/json/demo/312.json"
      initializeExperiment()
  $('#load-icon').hide()
  $('#slow-load').hide()
  $('#success-load').show()
  $('#load-btn').click initializeExperiment


initializeExperiment = ->
  $('#jspsych-target').html ''
  console.log 'INITIALIZE EXPERIMENT'

  #  ======================== #
  #  ========= TEXT ========= #
  #  ======================== #

  # These functions will be executed by the jspsych plugin that
  # they are passed to. String interpolation will use the values
  # of global variables defined in this file at the time the function
  # is called.


  text =
    debug: -> if DEBUG then "`DEBUG`" else ''

  # ================================= #
  # ========= BLOCK CLASSES ========= #
  # ================================= #

  class Block
    constructor: (config) ->
      _.extend(this, config)
      @_block = this  # allows trial to access its containing block for tracking state
      if @_init?
        @_init()

  class TextBlock extends Block
    type: 'text'
    cont_key: []

  class ButtonBlock extends Block
    type: 'button-response'
    is_html: true
    choices: ['Continue']
    button_html: '<button class="btn btn-primary btn-lg">%choice%</button>'


  class QuizLoop extends Block
    loop_function: (data) ->
      console.log 'data', data
      for c in data[data.length].correct
        if not c
          return true
      return false

  class MouselabBlock extends Block
    type: 'mouselab-mdp'
    playerImage: 'static/images/spider.png'
    # moveDelay: PARAMS.moveDelay
    # clickDelay: PARAMS.clickDelay
    # moveEnergy: PARAMS.moveEnergy
    # clickEnergy: PARAMS.clickEnergy
    lowerMessage: """
      <b>Clicking on a node reveals its value for a $1 fee.<br>
      Move with the arrow keys.</b>
    """
    
    _init: ->
      _.extend(this, STRUCTURE)
      @trialCount = 0



  #  ============================== #
  #  ========= EXPERIMENT ========= #
  #  ============================== #

  img = (name) -> """<img class='display' src='static/images/#{name}.png'/>"""
  
  nodeValuesDescription = switch PARAMS.variance
    when "6_6_6" then """
      A node can have value -10, -5, 5, or 10. All values are equally likely.
    """
    when "2_4_24" then """
      The more steps it takes to reach a node, the more variable its value
      tends to be: The value of a node you can reach in **one** step is equally
      likely to be **-4, -2, 2, or 4**. The value of a node you can reach in **two**
      steps is equally likely to be **-8, -4, 4, or 8**. Finally,  the value of a
      node you can reach in **three** steps is equally likely to be **-48, -24, 24,
      or 48**.
    """
    when "24_4_2" then """
      The more steps it takes to reach a node, the less variable its value
      tends to be: The value of a node you can reach in **one** step is equally
      likely to be **-48, -24, 24, or 48**. The value of a node you can reach in
      **two** steps is equally likely to be **-8, -4, 4, or 8**. Finally,  the value
      of a node you can reach in **three** steps is equally likely to be  -4, -2,
      2, or 4.
    """
        
  # instruct_loop = new Block
  #   timeline: [instructions, quiz]
  #   loop_function: (data) ->
  #     for c in data[1].correct
  #       if not c
  #         return true  # try again
  #     psiturk.finishInstructions()
  #     psiturk.saveData()
  #     return false

  fullMessage = ""
  reset_score = new Block
    type: 'call-function'
    func: ->
      SCORE = 0

  divider = new TextBlock
    text: ->
      SCORE = 0
      "<div style='text-align: center;'> Press <code>space</code> to continue.</div>"

  
   divider_training_test  = new TextBlock
    text: ->
      SCORE = 0
      "<div style='text-align: center;'> Congratulations! You have completed the training block. <br/>      
       <br/> Press <code>space</code> to start the test block.</div>"

   test_block_intro  = new TextBlock
    text: ->
      SCORE = 0        
      markdown """ 
      <h1>Test block</h1>
     Welcome to the test block! Here, you can use what you have learned to earn a bonus. Concretely, #{bonus_text('long')} <br/> To thank you for your work so far, we'll start you off with **$50**.
      Good luck! 
      <div style='text-align: center;'> Press <code>space</code> to continue. </div>
      """
    
    
   divider_intro_training  = new TextBlock
    text: ->
      SCORE = 0
      "  <h1>Training</h1>  Congratulations! You have completed the instructions. Next, you will enter a training block where you can practice planning 10 times. After that, you will enter test block where you can use what you have learned to earn a bonus. <br/> Press <code>space</code> to start the training block."

   divider_pretest_training  = new TextBlock
    text: ->
      SCORE = 0
      "<h1>Training block</h1> <div style='text-align: center;'> You will now enter a training block where you can practice playing Web of Cash some more. After that, there will be a test block where you can use what you have learned to earn a bonus. <br/> Press <code>space</code> to start the training block.</div>"

        
        
  train_basic1 = new TextBlock
    text: ->
      SCORE = 0
      markdown """
      <h1> Web of Cash </h1>

      In this HIT, you will play a game called *Web of Cash*. You will guide a
      money-loving spider through a spider web. When you land on a gray circle
      (a ***node***) the value of the node is added to your score.

      You will be able to move the spider with the arrow keys, but only in the direction
      of the arrows between the nodes. The image below shows the web that you will be navigating when the game starts.

     <img class='display' style="width:50%; height:auto" src='static/images/web-of-cash-unrevealed.png'/>

    <div align="center">Press <code>space</code> to proceed.</div>
    """
    #lowerMessage: 'Move with the arrow keys.'
    #stateDisplay: 'never'
    #timeline: getTrials 0
    
#   train_basic2 = new MouselabBlock
#    blockName: 'train_basic2'
#    stateDisplay: 'always'
#    prompt: ->
#      markdown """
#      ## Some nodes are more important than others

      #{nodeValuesDescription} Please take a look at the example below to see what this means.

#      Try a few more rounds now!
#    """
#    lowerMessage: 'Move with the arrow keys.'
#    timeline: getTrials 5

  
#  train_hidden = new MouselabBlock
#    blockName: 'train_hidden'
#    stateDisplay: 'never'
#    prompt: ->
#      markdown """
#      ## Hidden Information
#
#      Nice job! When you can see the values of each node, it's not too hard to
#      take the best possible path. Unfortunately, you can't always see the
#      value of the nodes. Without this information, it's hard to make good
#      decisions. Try completing a few more rounds.
#    """
#    lowerMessage: 'Move with the arrow keys.'
#    timeline: getTrials 5

#  train_inspector = new MouselabBlock
#    blockName: 'train_inspector'
    # special: 'trainClick'
#    stateDisplay: 'click'
#    stateClickCost: 0
#    prompt: ->
#      markdown """
#      ## Node Inspector

#      It's hard to make good decision when you can't see what you're doing!
#      Fortunately, you have access to a ***node inspector*** which can reveal
#      the value of a node. To use the node inspector, simply click on a node.
#      **Note:** you can only use the node inspector when you're on the first
#      node.

#      Trying using the node inspector on a few nodes before making your first
#      move.
#    """
#    # but the node inspector takes some time to work and you can only inspect one node at a time.
#    timeline: getTrials 1
    # lowerMessage: "<b>Click on the nodes to reveal their values.<b>"


#  train_inspect_cost = new MouselabBlock
#    blockName: 'train_inspect_cost'
#    stateDisplay: 'click'
#    stateClickCost: PARAMS.inspectCost
#    prompt: ->
#      markdown """
#      ## The price of information
#
#      You can use node inspector to gain information and make better
#      decisions. But, as always, there's a catch. Using the node inspector
#      costs $#{PARAMS.inspectCost} per node. To maximize your score, you have
#      to know when it's best to gather more information, and when it's time to
#      act!
#    """
#    timeline: getTrials 1


  bonus_text = (long) ->
    # if PARAMS.bonusRate isnt .01
    #   throw new Error('Incorrect bonus rate')
    s = "**you will earn 1 cent for every $5 you make in the game.**"
    if long
      s += " For example, if your final score is $1000, you will receive a bonus of $2."
    return s


#  train_final = new MouselabBlock
#    blockName: 'train_final'
#    stateDisplay: 'click'
#    stateClickCost: PARAMS.inspectCost
#    prompt: ->
#      markdown """
#      ## Earn a Big Bonus

#     Nice! You've learned how to play *Web of Cash*, and you're almost ready
#      to play it for real. To make things more interesting, you will earn real
#      money based on how well you play the game. Specifically,
#      #{bonus_text('long')}

#      These are the **final practice rounds** before your score starts counting
#      towards your bonus.
#    """
#    lowerMessage: fullMessage
#    timeline: getTrials 5


#  train = new Block
#    training: true
#    timeline: [
#      train_basic1
#       divider    
#      train_basic2    
#      divider
#      train_hidden
#      divider
#      train_inspector
#       divider
#      train_inspect_cost
#      divider
#       train_final
#    ]


    
  quiz = new Block
    preamble: -> markdown """
      # Quiz

    """
    type: 'survey-multi-choice'
    questions: [
      "What is the range of node values in the first step?"
      "What is the range of node values in the last step?"
      "What is the cost of clicking?"
      "How much REAL money do you earn?"
    ]
    options: [
      ['$-4 to $4', '$-8 to $8', '$-48 to $48'],
      ['$-4 to $4', '$-8 to $8', '$-48 to $48'],
      ['$0', '$1', '$8', '$24'],    
      ['1 cent for every $1 you make in the game',
       '1 cent for every $5 you make in the game',
       '5 cents for every $1 you make in the game',
       '5 cents for every $10 you make in the game']
    ]

  pre_test_intro = new TextBlock
    text: ->
      SCORE = 0
      #prompt: ''
      #psiturk.finishInstructions()
      markdown """
      ## Node Inspector

      It's hard to make good decision when you can't see what you will get!
      Fortunately, you will have access to a ***node inspector*** which can reveal
      the value of a node. To use the node inspector, simply ***click on a node***. The image below illustrates how this works, and you can try this out on the **next** screen. 

      **Note:** you can only use the node inspector when you're on the first
      node. 

      <img class='display' style="width:50%; height:auto" src='static/images/web-of-cash.png'/>

      One more thing: **You must spend *at least* 7 seconds on each round.**
      If you finish a round early, you'll have to wait until 7 seconds have
      passed.      

      <div align="center"> Press <code>space</code> to continue. </div>
        
    """
       
        
  pre_test = new MouselabBlock
    minTime: 7
    show_feedback: false
    blockName: 'pre_test'
    stateDisplay: 'click'
    stateClickCost: PARAMS.inspectCost
    timeline: switch
      when SHOW_PARTICIPANT then DEMO_TRIALS
      when DEBUG then TRIALS.slice(6, 7)
      else getTrials 1
    startScore: 50        

        
  training = new MouselabBlock
    minTime: 7
    show_feedback: with_feedback
    blockName: 'training'
    stateDisplay: 'click'
    stateClickCost: PARAMS.inspectCost
    timeline: switch
      when SHOW_PARTICIPANT then DEMO_TRIALS
      when DEBUG then TRIALS.slice(6, 8)
      else getTrials 10
    startScore: 50
        
        
  post_test = new MouselabBlock
    minTime: 7
    show_feedback: false
    blockName: 'test'
    stateDisplay: 'click'
    stateClickCost: PARAMS.inspectCost
    timeline: switch
      when SHOW_PARTICIPANT then DEMO_TRIALS
      when DEBUG then TRIALS.slice(6, 8)
      else getTrials 20
    startScore: 50
    
    
    
  verbal_responses = new Block
    type: 'survey-text'
    preamble: -> markdown """
        # Please answer these questions

      """

    questions: [
        'How did you decide where to click?'
        'How did you decide where NOT to click?'
        'How did you decide when to stop clicking?'
        'Where were you most likely to click at the beginning of each round?'
        'Can you describe anything else about your strategy?'
    ]
    button: 'Finish'

  # TODO: ask about the cost of clicking
  finish = new Block
    type: 'survey-text'
    preamble: -> markdown """
        # You've completed the HIT

        Thanks for participating. We hope you had fun! Based on your
        performance, you will be awarded a bonus of
        **$#{calculateBonus().toFixed(2)}**.

        Please briefly answer the questions below before you submit the HIT.
      """

    questions: [
      'What did you learn?'    
      'Was anything confusing or hard to understand?'
      'What is your age?'
      'Additional coments?'
    ]
    button: 'Submit HIT'

  talk_demo = new Block
    timeline: [
      # new MouselabBlock
      #   lowerMessage: 'Move with the arrow keys.'
      #   stateDisplay: 'always'
      #   prompt: null
      #   stateClickCost: PARAMS.inspectCost
      #   timeline: getTrials 3

      divider

      new MouselabBlock
        stateDisplay: 'click'
        prompt: null
        stateClickCost: PARAMS.inspectCost
        timeline: TRIALS.slice(10,14)
    ]


  experiment_timeline = switch
    when SHOW_PARTICIPANT then [
      test
    ]
    when DEBUG then [
      train_basic1
      pre_test_intro
      pre_test
      divider_pretest_training    
      training
      divider_training_test
      test_block_intro
      post_test
      quiz
      verbal_responses
      finish
    ]
    when TALK then [
      talk_demo
    ]
    else [
      train_basic1
      #train_inspector
      #train_inspect_cost
      #instructions1    
      pre_test_intro
      pre_test
      divider_pretest_training    
      training
      divider_training_test
      test_block_intro
      post_test
      quiz
      verbal_responses
      finish
      ]

  # ================================================ #
  # ========= START AND END THE EXPERIMENT ========= #
  # ================================================ #

  # bonus is the total score multiplied by something
  calculateBonus = ->
    bonus = SCORE * PARAMS.bonusRate
    bonus = (Math.round (bonus * 100)) / 100  # round to nearest cent
    return Math.max(0, bonus)
  

  reprompt = null
  save_data = ->
    psiturk.saveData
      success: ->
        console.log 'Data saved to psiturk server.'
        if reprompt?
          window.clearInterval reprompt
        psiturk.computeBonus('compute_bonus', psiturk.completeHIT)
      error: -> prompt_resubmit


  prompt_resubmit = ->
    $('#jspsych-target').html """
      <h1>Oops!</h1>
      <p>
      Something went wrong submitting your HIT.
      This might happen if you lose your internet connection.
      Press the button to resubmit.
      </p>
      <button id="resubmit">Resubmit</button>
    """
    $('#resubmit').click ->
      $('#jspsych-target').html 'Trying to resubmit...'
      reprompt = window.setTimeout(prompt_resubmit, 10000)
      save_data()

  jsPsych.init
    display_element: $('#jspsych-target')
    timeline: experiment_timeline
    # show_progress_bar: true

    on_finish: ->
      if DEBUG
        jsPsych.data.displayData()
      else
        psiturk.recordUnstructuredData 'final_bonus', calculateBonus()
        save_data()

    on_data_update: (data) ->
      console.log 'data', data
      psiturk.recordTrialData data

