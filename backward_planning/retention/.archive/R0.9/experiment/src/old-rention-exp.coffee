###
experiment.coffee
Fred Callaway

Demonstrates the jsych-mdp plugin

###
# coffeelint: disable=max_line_length, indentation


# Globals.
psiturk = new PsiTurk uniqueId, adServerLoc, mode
isIE = false || !!document.documentMode
TRIALS = undefined
THRESHOLDS = undefined
DIRECTIONS = ["down","right","up","left"]

TEST_TRIALS = undefined
TRAIN_TRIALS = undefined
TEST_IDX = undefined
N_TEST = 6
N_TRAIN = 10
N_TRIALS = 16
SCORE = 0
STRUCTURE = undefined
calculateBonus = undefined

train = undefined

### 
TODO
- define trial_i
- object-level PRs
- demo
###

# if 'hidden' in document
#   document.addEventListener("visibilitychange", onchange);
# else if 'mozHidden' in document
#   document.addEventListener("mozvisibilitychange", onchange);
# else if 'webkitHidden' in document
#   document.addEventListener("webkitvisibilitychange", onchange);
# else if 'msHidden' in document
  # document.addEventListener("msvisibilitychange", onchange);


if DEBUG
  0
  # N_TEST = 1
  # N_TRAIN = 1
  # N_TRIALS = 2
# because the order of arguments of setTimeout is awful.
delay = (time, func) -> setTimeout func, time

# $(window).resize -> checkWindowSize 920, 720, $('#jspsych-target')
# $(window).resize()

if isIE
  $('#jspsych-target').hide()
  $('#IE_error').show()
  # document.getElementById("IE_error").style.display = "block"
else
# $(document).ready ->
    $(window).on 'load', ->
    # Load data and test connection to server.
    M= -> $('#failLoad').show()
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
    if SHOW_PARTICIPANT_DATA
      TRIALS = loadJson "static/json/data/1B.0/stimuli/#{COST_LEVEL}_cost.json"
    else
      TRIALS = loadJson "static/json/rewards_#{PARAMS.info_cost.toFixed(2)}.json"
      STRUCTURE = loadJson "static/json/structure.json"
      THRESHOLDS = loadJson "static/json/thresholds_#{COST_LEVEL}_cost.json"    
      console.log 'STRUCTURE', STRUCTURE
      console.log 'TRIALS', TRIALS
    condition_nr = condition % nrConditions
    # PARAMS=
    #   PR_type: conditions.PRType[condition_nr]
    #   feedback: conditions.PRType[condition_nr] != "none"
    #   info_cost: conditions.infoCost[condition_nr]
    #   message:  conditions.messageType[condition_nr]
    #   frequencyOfFB: conditions.frequencyOfFB[condition_nr]
    #   condition: condition_nr
    #   start_time: new Date
    
    #idx = _.shuffle (_.range N_TRIALS)
    #train_idx = idx[...N_TRAIN]
    #TEST_IDX = idx[N_TRAIN...]    
    #TRAIN_TRIALS = (TRIALS[i] for i in train_idx)
    #TEST_TRIALS = (TRIALS[i] for i in TEST_IDX)
    TRAIN_TRIALS = _.shuffle TRIALS.train
    TEST_TRIALS = _.shuffle TRIALS.test

    if DEBUG
      TRAIN_TRIALS = TRIALS

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
  $('#jspsych-target').html('')
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

    return_window: ->
      cutoff = new Date (RETURN_TIME.getTime() + 1000 * 60 * 60 * PARAMS.delay_window)
      tomorrow = if RETURN_TIME.getDate() > (new Date).getDate() then 'tomorrow' else ''
      return """
        <b>#{tomorrow}
        between #{format_time RETURN_TIME}
        and #{format_time cutoff}</b>
      """


    feedback: ->
      if STAGE2
        return []
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

              <b> In the first #{N_TRAIN} rounds, an expert will demonstrate
              optimal flight planning.</b> In the remaining #{N_TEST} rounds,
              you will make your own choices.
              """
            ]
        else if PARAMS.message == "simple"
            [markdown """
              # Instructions

              <b>You will receive feedback about your planning. This feedback will
              help you learn how to make better decisions.</b> After each flight, if
              you did not plan optimally, a feedback message will apear.

              In the example below, there is a 26 second timeout penalty.
              <b>The duration of the timeout penalty is proportional to how
              poorly you planned your route:</b> the more money you could have
              earned from observing more/less values and/or choosing a better
              route, the longer the delay. <b>If you perform optimally, no
              feedback will be shown and you can proceed immediately.</b> The
              example message here is not necessarily representative of the
              feedback you'll receive.

              This feedback will be presented after each of the first
              #{N_TRAIN} rounds; during the final #{N_TEST} rounds,
              no feedback will be presented.

              #{img('task_images/Slide4_simple.png')}
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

              In the example below, there is a 6 second timeout penalty. If
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

              #{img('task_images/Slide4_neutral2.png')}
              """
            ]
      else if PARAMS.message == "full"
            [markdown """
              # Instructions

              <b>You will receive feedback about your planning. This feedback
              will help you learn how to make better decisions.</b> After each
              flight a feedback message will apear. This message will tell you
              two things:

              1. Whether you observed too few relevant values or if you observed
                 irrelevant values (values of locations that you can't fly to).
              2. Whether you flew along the best route given your current location and
                 the information you had about the values of other locations.

              This feedback will be presented after each of the first
              #{N_TRAIN} rounds; during the final #{N_TEST} rounds,
              no feedback will be presented.

              If you observe too few relevant values, the message will say,
              "You should have gathered more information!"; if you observe too
              many values, it will say "You should have gathered less
              information!"; and the image below shows the message you will
              see when you collected the right information but used it
              incorrectly.

              #{img('task_images/Slide4_neutral.png')}
              """
            ]
      else []

    constantDelay: ->
      if PARAMS.PR_type != "none"
        ""
      else
        "<b>Note:</b> there will be short delays after taking some flights."

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
    _init: ->
      _.extend(this, STRUCTURE)
      @trialCount = 0



  #  ============================== #
  #  ========= EXPERIMENT ========= #
  #  ============================== #

  debug_slide = new Block
    type: 'html'
    url: 'test.html'

  check_code = new Block
    type: 'secret-code'
    code: 'elephant'

  check_returning = do ->
    console.log 'worker', uniqueId
    worker_id = uniqueId.split(':')[0]
    stage1 = (loadJson 'static/json/stage1.json')[worker_id]
    if stage1?
      console.log 'stage1.return_time', stage1.return_time
      return_time = new Date stage1.return_time
      console.log 'return_time', return_time

      if getTime() > return_time
        # Redefine test trials to match breakdown established in stage 1.
        TEST_TRIALS = (TRIALS[i] for i in stage1.test_idx)
        SCORE += stage1.score

        return new Block
          type: 'button-response'
          is_html: true
          choices: ['Continue']
          button_html: '<button id="return-continue" class="btn btn-primary btn-lg">%choice%</button>'
          stimulus: -> markdown """
            # Welcome back

            Thanks for returning to complete Stage 2! Your current bonus is
            **$#{calculateBonus().toFixed(2)}**. In this stage you'll have #{N_TEST} rounds to
            increase your bonus.

            Before you begin, you will review the instructions and take another
            quiz.
          """
      else
        return new Block
          type: 'text'
          cont_key: [null]
          text: -> markdown """
            # Stage 2 not ready yet

            You need to wait #{PARAMS.delay_hours} hours after completing Stage 1 before
            you can begin Stage 2. You can begin the HIT at
            #{format_time(return_time)} on #{format_date(return_time)}
          """
          # **If you return the HIT, you may not be able to take it again later.**
          # Please leave the HIT open until it is time for you to complete Stage 2.
    else
      return new Block
        type: 'text'
        cont_key: [null]
        text: -> markdown """
          # Stage 1 not completed

          We can't find you in our database. This is the second part of a two-part
          experiment. If you did not complete the first stage, please
          return this HIT. If you did complete Stage 1, please email
          cocosci.turk@gmail.com to report the error.
        """

  retention_instruction = new Block
    type: 'button-response'
    is_html: true
    choices: ['Continue']
    button_html: '<button class="btn btn-primary btn-lg">%choice%</button>'
    stimulus: ->
      markdown """
      # You are beginning a two-part experiment

      This experiment has two stages which you will complete in separate HITs.
      The total base payment for both hits is $1.75, plus a **performance-dependent
      bonus** of up to $3.50 ($2.50 is a typical bonus).

      Stage 1 takes about 15 minutes, and you will receive $0.75 when you
      complete it. You will complete Stage 2 in a second HIT.
      You can begin the second HIT #{text.return_window()}.
      If you do not begin the HIT within this time frame, you will not receive the
      second base payment or any bonus.

      Upon completing Stage 2, you will receive $1.00 plus your bonus of
      up to $3.50.<br>**By completing both stages, you can make up to
      $5.25**.

      <div class="alert alert-warning">
        Only continue if you can complete the second (~10 minute) HIT which
        which will be available #{text.return_window()}.
      </div>
    """

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
        money you earn or lose in a location, you have to click on it.</b>

        You can uncover the value of as many or as few locations as you wish before the first flight.
        But <b>once you move the airplane to a new location, you can no longer collect any additional information.</b>

        #{img('task_images/Slide1.png')}
        
        To navigate the airplane, use the arrows (the example above is non-interactive).
        Click "Next" to proceed.
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
        1. You must spend at least #{MIN_TIME} seconds on each round. A countdown timer
           will show you how much more time you must spend on the round. You
           won’t be able to proceed to the next round before the countdown has
           finished, but you can take as much time as you like afterwards.
        2. </b>You will earn <u>real money</u> for your flights.</b>
           Specifically, for every $1 you earn in the game, we will add 1
           cent to your bonus. Please note that each and every one of the
           #{N_TRIALS} rounds counts towards your bonus.

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
      ] .concat (if (not STAGE2) & PARAMS.PR_type != "none" & PARAMS.PR_type != "demonstration" then [
        "What does the feedback teach you?"
    ] else [])
    options: [
      ['True', 'False']
      ['$0.01', '$0.05', '$0.10', '$0.25', '$1.00','$1.25', '$1.50', '$2.50', '$2.95', '$3.50', '$3.95', '$4.00','$10.00']
      ['At most 1', 'At most 5', 'At most 10', 'At most 15', 'As many or as few as I wish']
      ['1% of my best score on any round'
       '1 cent for every $1 I earn in each round'
       '10% of my best score on any round'
       '10% of my score on a random round']
    ] .concat (if STAGE2 then []
    else if PARAMS.PR_type == "objectLevel" then [[
      'Whether I chose the move that was best.'
      'The length of the delay is based on how much more money I could have earned.'
      'All of the above.']]
    else if PARAMS.PR_type != "none" then [[
      'Whether I observed the rewards of relevant locations.'
      'Whether I chose the move that was best according to the information I had.'
      'The length of the delay is based on how much more money I could have earned by planning and deciding better.'
      'All of the above.']]
    else [])
    required: [true, true, true, true, true]
    correct: [
      'True'
      fmtMoney PARAMS.info_cost
      'As many or as few as I wish'
      '1 cent for every $1 I earn in each round'
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
    leftMessage: -> "Round: #{TRIAL_INDEX}/#{N_TRAIN}"
    demonstrate: PARAMS.PR_type is "demonstration"   
    timeline: TRAIN_TRIALS

  console.log 'train', train
  
  test = new Block
    leftMessage: -> 
      if STAGE2 
        "Round: #{TRIAL_INDEX}/#{N_TEST}"
      else
        "Round: #{TRIAL_INDEX - N_TRAIN}/#{N_TEST}"
    timeline: do ->
      tl = []
      if PARAMS.feedback and not STAGE2
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
        timeline: TEST_TRIALS
      return tl

  console.log 'test', test
    
      
        
  ask_email = new Block
    type: 'survey-text'
    preamble: -> markdown """
        # You've completed Stage 1

        So far, you've earned a bonus of **$#{calculateBonus().toFixed(2)}**.
        You will receive this bonus, along with the additional bonus you earn 
        in Stage 2 when you complete the second HIT. If you don't complete
        the second HIT, you will give up the bonus you have earned.

        The HIT for Stage 2 will have the title "Part 2 of two-part decision-making experiment"
        Remember, you must begin the HIT #{text.return_window()}.
        **Note:** The official base pay on mTurk will be $0.01;
        you'll receive the $1 base pay for Stage 2 as part of your bonus 
        (in addition to the bonus you earn).
      """

    questions: ['If you would like a reminder email, you can optionally enter it here.']
    button: 'Submit HIT'

  if STAGE1
    finish = new Block
        type: 'button-response'
        stimulus: ->     
            markdown """
            # You've completed Stage 1

            Remember to come back #{text.return_window()} to complete Stage 2.
            The HIT will be titled "Part 2 of two-part decision-making
            experiment". **Note:** The official base pay on mTurk will be $0.01;
            you'll receive the $1 base pay for Stage 2 as part of your bonus 
            (in addition to the bonus you earn).

            So far, you've earned a bonus of **$#{calculateBonus().toFixed(2)}**.
            You will receive this bonus, along with the additional bonus you earn 
            in Stage 2 when you complete the second HIT. If you don't complete
            the second HIT, you give up the bonus you have already earned.
            """
        is_html: true
        choices: ['Submit HIT']
        button_html: '<button class="btn btn-primary btn-lg">%choice%</button>'            
  else
    finish = new Block
        type: 'survey-text'
        preamble: ->
            markdown """
            # You've completed the HIT

            Thanks for participating. We hope you had fun! Based on your
            performance, you will be awarded a bonus of
            **$#{calculateBonus().toFixed(2)}**.

            Please briefly answer the questions below before you submit the HIT.
            """

        questions: [
            'How did you go about planning the route of the airplane?'
            'Did you learn anything about how to plan better?'
            'How old are you?'
            'Which gender do you identify with?' 
        ]
        rows: [4,4,1,1]
        button: 'Submit HIT'            
    

  ppl = new Block
    type: 'webppl'
    code: 'globalStore.display_element.html(JSON.stringify(flip()))'
    
  if DEBUG
    experiment_timeline = [
      # train
      # test
      # check_returning
      # check_code
      train
      test
      finish
      # ppl
    ]
  else
    experiment_timeline = do ->
      tl = []
      if STAGE1
        tl.push retention_instruction
      if STAGE2
        tl.push check_returning
      tl.push instruct_loop
      unless STAGE2
        tl.push train
      unless STAGE1
        tl.push test
      if STAGE1
        tl.push ask_email
      else
        tl.push finish
      return tl


  # ================================================ #
  # ========= START AND END THE EXPERIMENT ========= #
  # ================================================ #

  calculateBonus = (final=false) ->
    # data = jsPsych.data.getTrialsOfType 'mouselab-mdp'
    # score = sum (_.pluck data, 'score')
    # console.log 'score', score
    bonus = (Math.max 0, SCORE) * PARAMS.bonus_rate
    bonus = (Math.round (bonus * 100)) / 100  # round to nearest cent
    return bonus

  # # bonus is the score on a random trial.
  # BONUS = undefined
  # calculateBonus = ->
  #   if BONUS?
  #     return BONUS
  #   data = jsPsych.data.getTrialsOfType 'mouselab-mdp'
  #   BONUS = 0.05 * Math.max 0, (_.sample data).score
  #   psiturk.recordUnstructuredData 'final_bonus', BONUS
  #   return BONUS
  

  reprompt = null
  save_data = ->
    psiturk.saveData
      success: ->
        console.log 'Data saved to psiturk server.'
        if reprompt?
          window.clearInterval reprompt
        psiturk.computeBonus('compute_bonus', psiturk.completeHIT)
      error: prompt_resubmit


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
      completion_data =
        score: SCORE
        bonus: calculateBonus()
        return_time: RETURN_TIME?.getTime()
        test_idx: TEST_IDX
      if DEBUG
        jsPsych.data.displayData()
        console.log 'completion_data', completion_data
      else

        psiturk.recordUnstructuredData 'completed', completion_data

        save_data()

    on_data_update: (data) ->
      console.log 'data', data
      psiturk.recordTrialData data
      