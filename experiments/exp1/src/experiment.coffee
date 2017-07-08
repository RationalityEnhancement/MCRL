###
experiment.coffee
Fred Callaway

Demonstrates the jsych-mdp plugin

###
# coffeelint: disable=max_line_length, indentation


# Globals.
psiturk = new PsiTurk uniqueId, adServerLoc, mode

isIE = false || !!document.documentMode
TEST_TRIALS = undefined
TRAIN_TRIALS = undefined
N_TEST = undefined
N_TRAIN = undefined
N_TRIALS = undefined
# because the order of arguments of setTimeout is awful.
delay = (time, func) -> setTimeout func, time

# $(window).resize -> checkWindowSize 920, 720, $('#jspsych-target')
# $(window).resize()

if isIE
  $('#jspsych-target').hide()
  $('#IE_error').show()
  # document.getElementById("IE_error").style.display = "block"

# $(document).ready ->
$(window).on 'load', ->
  # Load data and test connection to server.
  slowLoad = -> $('#failLoad').show()
  loadTimeout = delay 12000, slowLoad

  psiturk.preloadImages [
    'static/images/example1.png'
    'static/images/example2.png'
    'static/images/example3.png'
    'static/images/money.png'
    'static/images/plane.png'
    'static/images/spider.png'
  ]


  delay 300, ->
    console.log 'Loading data'
    expData = loadJson "static/json/#{COST_LEVEL}_cost.json"
    console.log 'expData', expData

    
    condition_nr = condition % nrConditions
    PARAMS=
      PR_type: conditions.PRType[condition_nr]
      feedback: conditions.PRType[condition_nr] != "none"
      info_cost: conditions.infoCost[condition_nr]
      message:  conditions.messageType[condition_nr]
      frequencyOfFB: conditions.frequencyOfFB[condition_nr]
      condition: condition_nr
      start_time: new Date
    
        
    # PARAMS.bonus_rate = .1
    trials = expData.blocks.standard
    TRAIN_TRIALS = trials[...6]
    TEST_TRIALS = trials[6...]
    N_TRAIN = TRAIN_TRIALS.length
    N_TEST = TEST_TRIALS.length
    N_TRIALS = N_TRAIN + N_TEST
    psiturk.recordUnstructuredData 'params', PARAMS
    psiturk.recordUnstructuredData 'experiment_nr', experiment_nr
    psiturk.recordUnstructuredData 'condition_nr', condition_nr

    if DEBUG or DEMO
      createStartButton()
    else
      console.log 'Testing saveData'
      ERROR = null
      psiturk.saveData
        error: ->
          console.log 'ERROR saving data.'
          ERROR = true
        success: ->
          console.log 'Data saved to psiturk server.'
          clearTimeout loadTimeout
          delay 500, createStartButton


createStartButton = ->
  if DEBUG
    initializeExperiment()
    return
  document.getElementById("loader").style.display = "none"
  document.getElementById("successLoad").style.display = "block"
  document.getElementById("failLoad").style.display = "none"
  $('#load-btn').click initializeExperiment


initializeExperiment = ->
  console.log 'INITIALIZE EXPERIMENT'
            
  msgType = 
    switch PARAMS.message
      when 'none' then '_noMsg'
      when 'simple' then '_simpleMsg'
      else ''


  #  ======================== #
  #  ========= TEXT ========= #
  #  ======================== #

  # These functions will be executed by the jspsych plugin that
  # they are passed to. String interpolation will use the values
  # of global variables defined in this file at the time the function
  # is called.

  text =
    debug: -> if DEBUG then "`DEBUG`" else ''

    feedback: ->
      if PARAMS.PR_type != "none"
        if PARAMS.PR_type == "objectLevel"
            [markdown """
              # Instructions

              <b>You will receive feedback about your planning. This feedback
              will help you learn how to make better decisions.</b> After each
              flight, if you did not make the best move, a feedback message
              will apear. This message will tell you whether you flew along
              the best route given your current location, and what the best
              move would have been.
              
              This feedback will be presented after each of the first
              #{N_TRAIN} rounds; during the final #{N_TEST} rounds,
              no feedback will be presented.

              In the example below, the best move was not taken. As a result,
              there is a 15 second timeout penalty.<b> The duration of the
              timeout penalty is proportional to how poor of a move you made:
              </b> the more money you could have earned, the longer the delay.
              <b> If you perform optimally, no feedback will be shown and you
              can proceed immediately.</b>

              #{img('task_images/Slide5.png')}

              """
            ]
        else if PARAMS.PR_type == "demonstration"
            [markdown """
              # Instructions

              <b>You will receive guidance about how to plan. This guidance
              will help you learn how to make better decisions.</b> The first
              #{N_TRAIN} rounds will demonstrate what optimal planning and
              flight paths look like. In the remaining #{N_TEST} rounds, you
              will make your own choices.
              """
            ]
        else if PARAMS.message == "simple"
            [markdown """
              # Instructions

              <b>You will receive feedback about your planning. This feedback will
              help you learn how to make better decisions.</b> After each flight, if
              you did not plan optimally, a feedback message will apear.

              In the example below, there is a 15 second timeout penalty. <b>The duration of the timeout penalty is
              proportional to how poorly you planned your route:</b> the more
              money you could have earned from observing more/less values
              and/or choosing a better route, the longer the delay. <b>If you
              perform optimally, no feedback will be shown and you can proceed
              immediately.</b> The example message here is not necessarily
              representative of the feedback you'll receive.

              This feedback will be presented after each of the first
              #{N_TRAIN} rounds; during the final #{N_TEST} rounds,
              no feedback will be presented.

              #{img('task_images/Slide4_simpleMsg.png')}
              """
            ]
        else
            [markdown """
              # Instructions

              <b>You will receive feedback about your planning. This feedback will
              help you learn how to make better decisions.</b> After each flight, if
              you did not plan optimally, a feedback message will apear. This message
              will tell you two things:

              1. Whether you observed too few relevant values or if you observed
                 irrelevant values (values of locations that you can't fly to).
              2. Whether you flew along the best route given your current location and
                 the information you had about the values of other locations.

              This feedback will be presented after each of the first
              #{N_TRAIN} rounds; during the final #{N_TEST} rounds,
              no feedback will be presented.

              In the example below, there is a 15 second timeout penalty. If
              you observed too few relevant values, the message would say,
              "You should have gathered more information!"; if you observed
              too many values, it would say "You should have gathered less
              information!". <b>The duration of the timeout penalty is
              proportional to how poorly you planned your route:</b> the more
              money you could have earned from observing more/less values
              and/or choosing a better route, the longer the delay. <b>If you
              perform optimally, no feedback will be shown and you can proceed
              immediately.</b> The example message here is not necessarily
              representative of the feedback you'll receive.

              #{img('task_images/Slide4' + msgType + '.png')}
              """
            ]
      else if PARAMS.message == "full"
            [markdown """
              # Instructions

              <b>You will receive feedback about your planning. This feedback will
              help you learn how to make better decisions.</b> After each flight a feedback message will apear. This message
              will tell you two things:

              1. Whether you observed too few relevant values or if you observed
                 irrelevant values (values of locations that you can't fly to).
              2. Whether you flew along the best route given your current location and
                 the information you had about the values of other locations.

              This feedback will be presented after each of the first
              #{N_TRAIN} rounds; during the final #{N_TEST} rounds,
              no feedback will be presented.

              In the example below, if
              you observed too few relevant values, the message would say,
              "You should have gathered more information!"; if you observed
              too many values, it would say "You should have gathered less
              information!". The example message here is not necessarily
              representative of the feedback you'll receive.

              #{img('task_images/Slide4_noPR.png')}
              """
            ]
      else []

    constantDelay: ->
      if PARAMS.PR_type != "none"
        ""
      else
        "Note: there will be short delays after taking some flights."

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
    cont_key: ['space']

  class QuizLoop extends Block
    loop_function: (data) ->
      console.log 'data', data
      for c in data[data.length].correct
        if not c
          return true
      return false

  class MDPBlock extends Block
    type: 'mouselab-mdp'
    # playerImage: 'static/images/spider.png'
    _init: -> @trialCount = 0


  #  ============================== #
  #  ========= EXPERIMENT ========= #
  #  ============================== #

  debug_slide = new Block
    type: 'html'
    url: 'test.html'


  instructions = new Block
    type: "instructions"
    pages: [
      markdown """
        # Instructions #{text.debug()}

        In this game, you are in charge of flying an aircraft. As shown below,
        you will begin in the central location. The arrows show which actions
        are available in each location. Note that once you have made a move you
        cannot go back; you can only move forward along the arrows. There are
        eight possible final destinations labelled 1-8 in the image below. On
        your way there, you will visit two intermediate locations. <b>Every
        location you visit will add or subtract money to your account</b>, and
        your task is to earn as much money as possible. <b>To find out how much
        money you earn or lose in a location, you have to click on it.</b> You
        can uncover the value of as many or as few locations as you wish.

        #{img('task_images/Slide1.png')}

        To navigate the airplane, use the arrows (the example above is non-interactive).
        You can uncover the value of a location at any time. Click "Next" to proceed.
      """

      markdown """
        # Instructions

        You will play the game for #{N_TRIALS} rounds. The value of
        every location will change from each round to the next. At the
        begining of each round, the value of every location will be hidden,
        and you will only discover the value of the locations you click on.
        The example below shows the value of every location, just to give you
        an example of values you could see if you clicked on every location.
        <b>Every time you click a circle to observe its value, you pay a fee
        of #{fmtMoney PARAMS.info_cost}.</b>

        #{img('task_images/Slide2_' + COST_LEVEL + '.png')}

        Each time you move to a
        location, your profit will be adjusted. If you move to a location with
        a hidden value, your profit will still be adjusted according to the
        value of that location. #{do text.constantDelay}
      """

    ] . concat (do text.feedback) .concat [

      markdown """
        # Instructions

        There are two more important things to understand:
        1. You must spend at least 45 seconds on each round. A countdown timer
           will show you how much more time you must spend on the round. You
           won’t be able to proceed to the next round before the countdown has
           finished, but you can take as much time as you like afterwards.
        2. </b>You will earn <u>real money</u> for your flights.</b>
           Specifically, one of the #{N_TRIALS} rounds will be chosen
           at random and you will receive 5% of your earnings in that round as
           a bonus payment.

        #{img('task_images/Slide3.png')}

         You may proceed to take an entry quiz, or go back to review the instructions.
      """
    ]
    show_clickable_nav: true

  quiz = new Block
    preamble: -> markdown """
      # Quiz
    """
    type: 'survey-multi-choice'  # note: I've edited this jspysch file
    questions: [
      "True or false: The hidden values will change each time I start a new round."
      "How much does it cost to observe each hidden value?"
      "How many hidden values am I allowed to observe in each round?"
      "How is your bonus determined?"
      ] .concat (if PARAMS.PR_type != "none" & PARAMS.PR_type != "demonstration" then [
        "What does the feedback teach you?"
    ] else [])
    options: [
      ['True', 'False']
      ['$0.01', '$0.05', '$1.00', '$2.50']
      ['At most 1', 'At most 5', 'At most 10', 'At most 15', 'As many or as few as I wish']
      ['10% of my best score on any round'
       '10% of my total score on all rounds'
       '5% of my best score on any round'
       '5% of my score on a random round']
    ] .concat (if PARAMS.PR_type == "objectLevel" then [[
       'Whether I chose the move that was best.'
       'The length of the delay is based on how much more money I could have earned.'
       'All of the above.']
    ] else if PARAMS.PR_type != "none" then [[
       'Whether I observed the rewards of relevant locations.'
       'Whether I chose the move that was best according to the information I had.'
       'The length of the delay is based on how much more money I could have earned by planning and deciding better.'
       'All of the above.']
    ] else [])
    required: [true, true, true, true, true]
    correct: [
      'True'
      fmtMoney PARAMS.info_cost
      'As many or as few as I wish'
      '5% of my score on a random round'
      'All of the above.'
    ]
    on_mistake: (data) ->
      alert """You got at least one question wrong. We'll send you back to the
               instructions and then you can try again."""


  instruct_loop = new Block
    timeline: [instructions, quiz]
    loop_function: (data) ->
      for c in data[1].correct
        if not c
          return true  # try again
      psiturk.finishInstructions()
      psiturk.saveData()
      return false


  # for t in BLOCKS.standard
  #   _.extend t, t.stim.env
  #   t.pseudo = t.stim.pseudo

  train = new MDPBlock
    demonstrate: PARAMS.PR_type is "demonstration"   
    timeline: _.shuffle TRAIN_TRIALS
  
  test = new Block
    timeline: do ->
      tl = []
      if PARAMS.feedback
        tl.push new TextBlock
          text: markdown """
            # No more feedback

            You are now entering a block without feedback. There will be no
            messages and no delays regardless of what you do, but your
            performance still affects your bonus.

            Press **space** to continue.
            """
      if PARAMS.PR_type is "demonstration"
        tl.push new TextBlock
          text: markdown """
            # Your turn
            
            This was the last demonstration from your teacher. Now it is your
            turn to decide which locations to inspect and where to fly to.

            Press **space** to continue.
            """        
                
      tl.push new MDPBlock
        feedback: false
        timeline: _.shuffle TEST_TRIALS
      return tl
    
      
        

  console.log 'test', test
  finish = new Block
    type: 'button-response'
    stimulus: -> markdown """
      # You've completed the HIT

      Thanks again for participating. We hope you had fun!

      Based on your performance, you will be
      awarded a bonus of **$#{calculateBonus().toFixed(2)}**.
      """
    is_html: true
    choices: ['Submit hit']
    button_html: '<button class="btn btn-primary btn-lg">%choice%</button>'


  if DEBUG
    experiment_timeline = [
      # instruct_loop
      train
      test
      finish
    ]
  else
    experiment_timeline = [
      instruct_loop
      train
      test
      finish
    ]


  # ================================================ #
  # ========= START AND END THE EXPERIMENT ========= #
  # ================================================ #

  # calculateBonus = ->
  #   if BONUS?
  #     return BONUS
  #   data = jsPsych.data.getTrialsOfType 'mouselab-mdp'
  #   bonus = mean (_.pluck data, 'score')
  #   bonus = (Math.round (bonus * 100)) / 100
  #   BONUS =  (Math.max 0, bonus) * PARAMS.bonus_rate
  #   psiturk.recordUnstructuredData 'final_bonus', BONUS
  #   return BONUS

  # bonus is the score on a random trial.
  BONUS = undefined
  calculateBonus = ->
    if BONUS?
      return BONUS
    data = jsPsych.data.getTrialsOfType 'mouselab-mdp'
    BONUS = 0.05 * Math.max 0, (_.sample data).score
    psiturk.recordUnstructuredData 'final_bonus', BONUS
    return BONUS
  

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
      