###
experiment.coffee
Fred Callaway

Demonstrates the jsych-mdp plugin

###
# coffeelint: disable=max_line_length, indentation



DEBUG = yes

if DEBUG
  console.log """
  X X X X X X X X X X X X X X X X X
   X X X X X DEBUG  MODE X X X X X
  X X X X X X X X X X X X X X X X X
  """
  condition = 0
else
  console.log """
  # =============================== #
  # ========= NORMAL MODE ========= #
  # =============================== #
  """

if mode is "{{ mode }}"
  DEMO = true
  condition = 0
  counterbalance = 0


# Globals.
psiturk = new PsiTurk uniqueId, adServerLoc, mode

BLOCKS = undefined
PARAMS = undefined

# because the order of arguments of setTimeout is awful.
delay = (time, func) -> setTimeout func, time

# $(window).resize -> checkWindowSize 920, 720, $('#jspsych-target')
# $(window).resize()

# $(document).ready ->
$(window).on 'load', ->
  # Load data and test connection to server.
  slowLoad = -> document.getElementById("failLoad").style.display = "block"
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
    expData = deepLoadJson "static/json/condition_#{condition}_#{counterbalance}.json"
    console.log 'expData', expData
    PARAMS = expData.params
    PARAMS.bonus_rate = .1
    PARAMS.start_time = Date(Date.now())
    BLOCKS = expData.blocks
    psiturk.recordUnstructuredData 'params', PARAMS

    if DEBUG or DEMO
      createStartButton()
      PARAMS.message = true
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

  N_TRIAL = BLOCKS.standard.length
  #  ======================== #
  #  ========= TEXT ========= #
  #  ======================== #

  # These functions will be executed by the jspsych plugin that
  # they are passed to. String interpolation will use the values
  # of global variables defined in this file at the time the function
  # is called.


  text =
    debug: -> if DEBUG then "`DEBUG`" else ''
    pseudo: ->
      switch PARAMS.pseudo_f
        when 'full'
          """
          The number of stars on a circle indicates the maximum amount of money
          you can earn if you pass through that circle.
          """
        when 'value'
          """
          The number of stars on a circle indicates the maximum amount of money
          you can earn starting from that circle.
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
    cont_key: ['space']

  class QuizLoop extends Block
    loop_function: (data) ->
      console.log 'data', data
      for c in data[data.length].correct
        if not c
          return true
      return false

  class GraphBlock extends Block
    type: 'mouselab-mdp'
    playerImage: 'static/images/spider.png'
    _init: -> @trialCount = 0


  #  ============================== #
  #  ========= EXPERIMENT ========= #
  #  ============================== #

  img = (name) -> """<img class='display' src='static/images/#{name}.png'/>"""

  instructions = new Block
    type: "instructions"
    pages: -> [
      markdown """
        # Introduction

        In this HIT, you will play a game called *Web of Cash*. You will guide
        a money-loving spider through a spider web, gaining or losing money
        for every move you make. Your goal is to maximize the profit for each
        round. Note the direction of the arrows: You can only travel right,
        never left. For each move, you can decide whether to go up or down.

        #{img 'example1'}

      """

      markdown """
        # Bonus Pay

        To make things more exciting, you will earn **real money** based on
        how well you do in the game. After you complete all #{N_TRIAL} rounds,
        we will calculate the average profit you made on all the trials. Your bonus
        will be #{PARAMS.bonus_rate * 100}% of that amount, up to
        a maximum of **$2.30**!

        
        #{img 'money'}
      """
      
      markdown """
        # Inspecting the Web

        In the previous example, the money you would make by crossing each
        arrow was shown on the arrow. However, in the real game, these numbers
        will not be shown when the round starts! Fortunately, **you can reveal the value of
        an arrow by clicking on it.**

        #{img 'example2'}

      """

      markdown """
        # Helpful Stars

        In some rounds, some circles will have stars on them. **You can click on
        those circles to reveal the number of stars**. These stars provide
        information that can **help you earn more money!** #{text.pseudo()}
        For example: in the image below, you can earn $18 starting from the
        position circled in blue by following the path indicated by the purple
        circles.

        #{img 'example3'}
      """

      markdown """
        # In case of technical difficulties

        We've tried our best to prevent any glitches, but no one's perfect! If
        something goes wrong during the experiment, you can always email
        fredcallaway@berkeley.edu. However, the fastest way to get reimbursed for
        your time is to fill out the following form. We suggest you copy down
        the URL now, just in case. Please include a short description of what
        happened and where you were in the experiment (e.g. what round number)
        Thanks!

        https://goo.gl/forms/CW0cAKyOHipFGXZE2
      """

      markdown """
        # Quiz

        Next up is a short quiz to confirm that you understand how to play
        *Web of Cash*. If you get any questions wrong, you'll be sent back to
        the instructions to review before taking the quiz again. Good luck!
      """
    ]
    show_clickable_nav: true


  quiz = new Block
    type: 'survey-multi-choice'  # note: I've edited this jspysch file
    preamble: -> markdown """
      # Quiz
    """
    questions: [
      """
        What will your bonus be based on?
      """
      # """
      #   What does it mean for an arrow to have -3 on it?
      # """
      """
        What does it mean for a location to have 7 stars on it?
      """
      """
        What does it mean when there is a question mark on an arrow?
      """
    ]
    options: [
      ['Profit', 'Stars', 'Both profit and stars']
      [
        'You will receive $70 for visiting that location'
        'You can earn a maximum of $7 after visiting that location'
        'You can earn a maximum of $7 in the entire round if you visit that location'
      ]
      [
        'You will receive no money for crossing that arrow'
        'You will receive a random amount of money for crossing that arrow'
        'You can click on the question mark to reveal the value of that arrow'
      ]

    ]
    required: [true, true, true]
    correct: [
      'Profit'
      'You can earn a maximum of $7 after visiting that location'
      'You can click on the question mark to reveal the value of that arrow'
    ]
    on_mistake: (data) ->
      alert """You got at least one question wrong. We'll send you back to review
               the instructions; then you can try again."""


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


  main = new GraphBlock
    timeline: BLOCKS.standard


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
      main
      finish
    ]
  else
    experiment_timeline = [
      instruct_loop
      main
      finish
    ]


  # ================================================ #
  # ========= START AND END THE EXPERIMENT ========= #
  # ================================================ #

  # bonus is the score on a random trial.
  BONUS = undefined
  calculateBonus = ->
    if BONUS?
      return BONUS
    data = jsPsych.data.getTrialsOfType 'mouselab-mdp'
    bonus = mean (_.pluck data, 'score')
    bonus = (Math.round (bonus * 100)) / 100
    BONUS =  (Math.max 0, bonus) * PARAMS.bonus_rate
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

