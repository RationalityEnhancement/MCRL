# coffeelint: disable=max_line_length, indentation
BLOCKS = undefined
TRIALS_TRAINING = undefined
TRIALS_TEST = undefined
DEMO_TRIALS = undefined
STRUCTURE_TEST = undefined
STRUCTURE_TRAINING = undefined
N_TRIAL = undefined
SCORE = 0
calculateBonus = undefined
getTrainingTrials = undefined
getTestTrials = undefined


DEBUG = false
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
#_.compose = _.flowRight
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
  condition: condition
  bonusRate: .002
  delay_hours: 24
  delay_window: 12
  branching: '312'
  with_feedback: with_feedback
  condition: CONDITION   
  startTime: Date(Date.now())
  variance: '2_4_24'    


RETURN_TIME = new Date (getTime() + 1000 * 60 * 60 * PARAMS.delay_hours)

MIN_TIME = 7

psiturk = new PsiTurk uniqueId, adServerLoc, mode

psiturk.recordUnstructuredData 'condition', CONDITION   
psiturk.recordUnstructuredData 'with_feedback', with_feedback



delay = (time, func) -> setTimeout func, time
# $(window).resize -> checkWindowSize 920, 720, $('#jspsych-target')
# $(window).resize()
slowLoad = -> $('slow-load')?.show()
loadTimeout = delay 12000, slowLoad


createStartButton = ->
  if DEBUG
    initializeExperiment()
    return
  document.getElementById("loader").style.display = "none"
  document.getElementById("successLoad").style.display = "block"
  document.getElementById("failLoad").style.display = "none"
  $('#load-btn').click initializeExperiment


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
        
    psiturk.recordUnstructuredData 'params', PARAMS

    if PARAMS.variance
      id = "#{PARAMS.branching}_#{PARAMS.variance}"
    else
      id = "#{PARAMS.branching}"
    STRUCTURE_TEST = loadJson "static/json/structure/31123.json"
    STRUCTURE_TRAINING = loadJson "static/json/structure/312.json"
    #TRIALS = loadJson "static/json/mcrl_trials/increasing.json"
    TRIALS_TEST = loadJson "static/json/rewards/31123_increasing1.json"
    console.log "loaded #{TRIALS_TEST?.length} test trials"
    TRIALS_TRAINING = loadJson "static/json/mcrl_trials/increasing.json"
    console.log "loaded #{TRIALS_TRAINING?.length} training trials"

    getTrainingTrials = do ->
      t = _.shuffle TRIALS_TRAINING
      idx = 0
      return (n) ->
        idx += n
        t.slice(idx-n, idx)

    getTestTrials = do ->
      t = _.shuffle TRIALS_TEST
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
          else
            return """

            """
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

        #_init: ->
          #_.extend(this, STRUCTURE)
        #  @trialCount = 0



      #  ============================== #
      #  ========= EXPERIMENT ========= #
      #  ============================== #

      img = (name) -> """<img class='display' src='static/images/#{name}.png'/>"""

    class QuizLoop extends Block
        loop_function: (data) ->
          console.log 'data', data
          for c in data[data.length].correct
            if not c
              return true
          return false

      # instruct_loop = new Block
      #   timeline: [instructions, quiz]
      #   loop_function: (data) ->
      #     for c in data[1].correct
      #       if not c
      #         return true  # try again
      #     psiturk.finishInstructions()
      #     psiturk.saveData()
      #     return false

    check_code = new Block
        type: 'secret-code'
        code: 'elephant'

    check_returning = do ->
        console.log 'worker', uniqueId
        if DEBUG
          worker_id = 'debugSRVTKD'
        else
          worker_id = uniqueId.split(':')[0]

        stage1 = (loadJson 'static/json/stage1.json')[worker_id]
        if stage1?
          console.log 'stage1.return_time', stage1.return_time
          return_time = new Date stage1.return_time
          console.log 'return_time', return_time    

          if getTime() > return_time
            # Redefine test trials to match breakdown established in stage 1.
            #TEST_TRIALS = (TRIALS[i] for i in stage1.test_idx)
            #SCORE += stage1.score

            return new Block
              type: 'button-response'
              is_html: true
              choices: ['Continue']
              button_html: '<button id="return-continue" class="btn btn-primary btn-lg">%choice%</button>'
              stimulus: -> markdown """
                # Welcome back

                Thanks for returning to complete Stage 2!

                After practicing on the simple version of Web of Cash in Stage 1, you can now use what you have learned to earn real money in the difficult version.

                Before you begin, let us give you a brief refresher of how the game works.
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
          # You are beginning a two-stage experiment

          This experiment has two stages which you will complete in separate HITs.
          The total base payment for both HITs is $2.00.

          Stage 1 takes about 5 minutes. It pays only  $0.10 but you can earn a performance-dependent but it makes you eligible
          to participate in Stage 2 where you can earn $1.90 in 10 minutes plus a performance-dependent
          bonus of up to $3.50 ($1.30 is a typical bonus). 
          You will complete Stage 2 in a second HIT which you can begin #{text.return_window()}.
          If you do not begin the HIT within this time frame, you will not receive the
          second base payment or any bonus.           

          By completing both stages, you can make up to
          $5.50, but if you don't complete Stage 2, you will lose your bonus from Stage 1 and the HIT would be a very bad deal for you.

          <div class="alert alert-warning">
            Please do <b>NOT<b> continue unless you are certain that you will complete the second HIT which
            which becomes available #{text.return_window()}. Completing only the first HIT would be a very bad deal for you (corresponding to a wage of $1.20/hour) and it would be bad for us too. You will be much better of if you complete both HITs (corresponding to a wage of about $15.20/hour.) and we need that for our experiment to work.
          </div>
        """

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
            you'll receive the $1.90 base pay for Stage 2 as part of your bonus 
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
          "<div style='text-align: left;'> Congratulations! You have completed the training block. <br/>      
           <br/> Press <code>space</code> to start the test block.</div>"

       test_block_intro  = new TextBlock
        text: ->
          SCORE = 0        
          markdown """ 
          <h1>Test block</h1>
         Welcome to the test block! Here, you can use what you have learned to earn a bonus. Concretely, #{bonus_text('long')} <br/> To thank you for your work so far, we'll start you off with **$100**.
          Good luck! 
          <div style='text-align: center;'> Press <code>space</code> to continue. </div>
          """


       #divider_intro_training  = new TextBlock
    #    text: ->
    #      SCORE = 0
    #      "  <h1>Training</h1>  Congratulations! You have completed the instructions. Next, you will enter a training block where you can practice planning 10 times. After that, you will enter a test block where you can use what you have learned to earn a bonus. <br/> Press <code>space</code> to start the training block."

       divider_pretest_training  = new TextBlock
        text: ->
          SCORE = 0
          "<h1>Training block</h1> 
    <p> The game you just played is quite complex and it can be rather difficult to get it right. To help you master it, we will now let you practice on a simplified version of this game 10 times. </p>

    <p> In the simplified version your goal is to find the most profitable route of an airplane across a network of airports. There will be only three steps but otherwise the game works just like the one you just played. </p>

    <p>After that, there will be a test block where you can use what you have learned to earn a bonus. </p>

    <br/> Press <code>space</code> to start the training block.</div>"



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
        s = "**you will earn 20 cent for every $100 you make in the game.**"
        if long
          s += " For example, if your final score is $500, you will receive a bonus of $1."
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

      pre_test_intro1 = new TextBlock
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

      pre_test_intro2 = new TextBlock
        text: ->
          SCORE = 0
          #prompt: ''
          #psiturk.finishInstructions()
          markdown """
          ## Get ready!

          You are about to play your first round of Web of Cash. You will notice that the web used in this game is larger than the example you saw in the previous pictures. But that is the only difference, and everything else works as described. Good luck!

          <div align="center"> Press <code>space</code> to continue. </div>

        """
      
      refresher1 = new TextBlock
        text: ->
          markdown """
          <h1> Refresher 1</h1>

          In this HIT, you will play a game called *Web of Cash*. You will guide a
          money-loving spider through a spider web. When you land on a gray circle
          (a ***node***) the value of the node is added to your score.

          You will be able to move the spider with the arrow keys, but only in the direction
          of the arrows between the nodes. The image below shows the web that you will be navigating when the game starts.

         <img class='display' style="width:50%; height:auto" src='static/images/web-of-cash-unrevealed.png'/>

        <div align="center">Press <code>space</code> to proceed.</div>
        """    

      refresher2 = new TextBlock
        text: ->
          markdown """
          <h1> Refresher 2</h1>

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
          when DEBUG then getTestTrials 1
          else getTestTrials 1
        startScore: 50        
        _init: ->
          _.extend(this, STRUCTURE_TEST)
          @trialCount = 0


      training = new MouselabBlock
        minTime: 7
        show_feedback: with_feedback
        blockName: 'training'
        stateDisplay: 'click'
        stateClickCost: PARAMS.inspectCost
        timeline: switch
          when SHOW_PARTICIPANT then DEMO_TRIALS
          when DEBUG then getTrainingTrials 2
          else getTrainingTrials 10
        startScore: 50
        _init: ->
          _.extend(this, STRUCTURE_TRAINING)
          @playerImage = 'static/images/plane.png'
          @trialCount = 0


      post_test = new MouselabBlock
        minTime: 7
        show_feedback: false
        blockName: 'test'
        stateDisplay: 'click'
        stateClickCost: PARAMS.inspectCost
        timeline: switch
          when SHOW_PARTICIPANT then DEMO_TRIALS
          when DEBUG then getTestTrials 10
          else getTestTrials 20
        startScore: 100
        _init: ->
          _.extend(this, STRUCTURE_TEST)
          @trialCount = 0


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
            timeline: getTestTrials 4
        ]


    #  experiment_timeline = switch
    #    when SHOW_PARTICIPANT then [
    #      test
    #    ]
    #    when DEBUG then [
    #      train_basic1
    #      pre_test_intro1
    #      pre_test_intro2
    #      pre_test
    #      divider_pretest_training    
    #      training
    #      divider_training_test
    #      test_block_intro
    #      post_test
    #      #quiz
    #      #verbal_responses
    #      finish
    #    ]
    #    when TALK then [
    #      talk_demo
    #    ]
    #    else [
    #      train_basic1
    #      pre_test_intro1
    #      pre_test_intro2
    #      pre_test
    #      divider_pretest_training    
    #      training
    #      divider_training_test
    #      test_block_intro
    #      post_test
    #      #quiz
    #      #verbal_responses
    #      finish
    #      ]

      if DEBUG
        experiment_timeline = do ->
          tl = []
          if STAGE1
            tl.push retention_instruction
          if STAGE2
            tl.push check_returning
            tl.push refresher1
            tl.push refresher2
          #tl.push instruct_loop
          unless STAGE2
            tl.push train_basic1
            tl.push pre_test_intro1
            tl.push pre_test_intro2
            tl.push pre_test     
            tl.push divider_pretest_training    
            tl.push training
          unless STAGE1
            tl.push test_block_intro
            tl.push post_test
          if STAGE1
            tl.push ask_email
          else
            tl.push finish
          return tl
      else
        experiment_timeline = do ->
          tl = []
          if STAGE1
            tl.push retention_instruction
          if STAGE2
            tl.push check_returning
            tl.push refresher1
            tl.push refresher2
          #tl.push instruct_loop
          unless STAGE2
            tl.push train_basic1
            tl.push pre_test_intro1
            tl.push pre_test_intro2
            tl.push pre_test     
            tl.push divider_pretest_training    
            tl.push training
          unless STAGE1
            tl.push test_block_intro
            tl.push post_test
          if STAGE1
            tl.push ask_email
          else
            tl.push finish
          return tl

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

