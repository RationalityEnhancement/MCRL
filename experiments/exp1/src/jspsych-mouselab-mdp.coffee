###
jspsych-mouselab-mdp.coffee
Fred Callaway

https://github.com/fredcallaway/Mouselab-MDP
###

# coffeelint: disable=max_line_length
mdp = undefined
OPTIMAL = undefined
TRIAL_INDEX = 1
jsPsych.plugins['mouselab-mdp'] = do ->


  PRINT = (args...) -> console.log args...
  NULL = (args...) -> null
  LOG_INFO = PRINT
  LOG_DEBUG = NULL

  # a scaling parameter, determines size of drawn objects
  SIZE = undefined
  DEMO_SPEED = 1000
  MOVE_SPEED = 500
  UNKNOWN = '__'
  TERM_ACTION = '__TERM_ACTION__'


  fabric.Object::originX = fabric.Object::originY = 'center'
  fabric.Object::selectable = false
  fabric.Object::hoverCursor = 'plain'

  if SHOW_PARTICIPANT_DATA
    OPTIMAL = loadJson "static/json/data/1B.0/traces/#{SHOW_PARTICIPANT_DATA}.json"
    DEMO_SPEED = 500
    MOVE_SPEED = 300
  else
    OPTIMAL = (loadJson 'static/json/optimal_policy.json')[COST_LEVEL]

  # =========================== #
  # ========= Helpers ========= #
  # =========================== #

  angle = (x1, y1, x2, y2) ->
    x = x2 - x1
    y = y2 - y1
    if x == 0
      ang = if y == 0 then 0 else if y > 0 then Math.PI / 2 else Math.PI * 3 / 2
    else if y == 0
      ang = if x > 0 then 0 else Math.PI
    else
      ang = if x < 0
        Math.atan(y / x) + Math.PI
      else if y < 0
        Math.atan(y / x) + 2 * Math.PI
      else Math.atan(y / x)
    return ang + Math.PI / 2

  polarMove = (x, y, ang, dist) ->
    x += dist * Math.sin ang
    y -= dist * Math.cos ang
    return [x, y]

  dist = (o1, o2) ->
    ((o1.left - o2.left) ** 2 + (o1.top - o2.top)**2) ** 0.5

  redGreen = (val) ->
    if val > 0
      '#080'
    else if val < 0
      '#b00'
    else
      '#888'

  round = (x) ->
    (Math.round (x * 100)) / 100

  checkObj = (obj, keys) ->
    if not keys?
      keys = Object.keys(obj)
    for k in keys
      if obj[k] is undefined
        console.log 'Bad Object: ', obj
        throw new Error "#{k} is undefined"
    obj

  KEYS = _.mapObject
    2: 'uparrow'
    0: 'downarrow',
    1: 'rightarrow',
    3: 'leftarrow',
    jsPsych.pluginAPI.convertKeyCharacterToKeyCode


# =============================== #
# ========= MouselabMDP ========= #
# =============================== #
  
  class MouselabMDP
    constructor: (config) ->
      {
        @display  # html display element

        @graph  # defines transition and reward functions
        @layout  # defines position of states
        @tree=null
        @initial  # initial state of player

        @stateLabels=null  # object mapping from state names to labels
        @stateDisplay='never'  # one of 'never', 'hover', 'click', 'always'
        @stateClickCost=PARAMS.info_cost  # subtracted from score every time a state is clicked
        @edgeLabels='reward'  # object mapping from edge names (s0 + '__' + s1) to labels
        @edgeDisplay='always'  # one of 'never', 'hover', 'click', 'always'
        @edgeClickCost=0  # subtracted from score every time an edge is clicked
        @trial_i=null
        @demonstrate=false

        @stateRewards=null
        
        @keys=KEYS  # mapping from actions to keycodes
        @trialIndex=TRIAL_INDEX  # number of trial (starts from 1)
        @playerImage='static/images/plane.png'
        SIZE=110  # determines the size of states, text, etc...

        leftMessage=null
        centerMessage='&nbsp;'
        rightMessage='Score: <span id=mouselab-score/>'
        lowerMessage="Navigate with the arrow keys."

        @minTime=(if DEBUG then 5 else 45)
        @feedback=true
      } = config

      @initial = 0
      @tree =  [ 
         [ 1, 5, 9, 13 ],
         [ 2 ],
         [ 3, 4 ],
         [],
         [],
         [ 6 ],
         [ 7, 8 ],
         [],
         [],
         [ 10 ],
         [ 11, 12 ],
         [],
         [],
         [ 14 ],
         [ 15, 16 ],
         [],
         [] ]
      @transition = [
       { '0': 1, '1': 5, '2': 9, '3': 13 },
       { '0': 2 },
       { '1': 3, '3': 4 },
       {},
       {},
       { '1': 6 },
       { '0': 8, '2': 7 },
       {},
       {},
       { '2': 10 },
       { '1': 12, '3': 11 },
       {},
       {},
       { '3': 14 },
       { '0': 15, '2': 16 },
       {},
       {} ]
      @layout =    [ [ 0, 0 ],
         [ 0, 1 ],
         [ 0, 2 ],
         [ 1, 2 ],
         [ -1, 2 ],
         [ 1, 0 ],
         [ 2, 0 ],
         [ 2, -1 ],
         [ 2, 1 ],
         [ 0, -1 ],
         [ 0, -2 ],
         [ -1, -2 ],
         [ 1, -2 ],
         [ -1, 0 ],
         [ -2, 0 ],
         [ -2, 1 ],
         [ -2, -1 ] ]
        

      if not leftMessage?
        leftMessage = "Round: #{TRIAL_INDEX}/#{N_TRIALS}"

      if @demonstrate
        lowerMessage = "This is a demonstration of optimal planning."

      if SHOW_PARTICIPANT_DATA
        @demonstrate = true
        lowerMessage = "Behavior of participant #{SHOW_PARTICIPANT_DATA}"

      console.log 'TRIAL NUMBER', @trial_i
      # _.extend this, config
      checkObj this
      @initial = "#{@initial}"
      @invKeys = _.invert @keys
      @data =
        delays: []
        planned_too_little: []
        planned_too_much: []
        information_used_correctly: []
        trial_i: @trial_i
        trialIndex: @trialIndex
        score: 0
        path: []
        rt: []
        actions: []
        actionTimes: []
        beliefs: []
        metaActions: []
        queries: {
          click: {
            state: {'target': [], 'time': []}
            edge: {'target': [], 'time': []}
          }
          mouseover: {
            state: {'target': [], 'time': []}
            edge: {'target': [], 'time': []}
          }
          mouseout: {
            state: {'target': [], 'time': []}
            edge: {'target': [], 'time': []}
          }
        }
        # clicks: []
        # clickTimes: []

      @leftMessage = $('<div>',
        id: 'mouselab-msg-left'
        class: 'mouselab-header'
        html: leftMessage).appendTo @display

      # @centerMessage = $('<div>',
      #   id: 'mouselab-msg-center'
      #   class: 'mouselab-header'
      #   html: centerMessage).appendTo @display
      timeMsg = if @demonstrate then '&nbsp;' else 'Time: <span id=mdp-time/>'
      @centerMessage = $('<div>',
        id: 'mouselab-msg-center'
        class: 'mouselab-header'
        html: timeMsg).appendTo @display

      @rightMessage = $('<div>',
        id: 'mouselab-msg-right',
        class: 'mouselab-header'
        html: rightMessage).appendTo @display
      @addScore 0
          
      @canvasElement = $('<canvas>',
        id: 'mouselab-canvas',
      ).attr(width: 500, height: 500).appendTo @display

      @lowerMessage = $('<div>',
        id: 'mouselab-msg-bottom'
        html: lowerMessage or '&nbsp'
      ).appendTo @display

      mdp = this
      LOG_INFO 'new MouselabMDP', this

      # feedback element
      $('#jspsych-target').append """
      <div id="mdp-feedback" class="modal">
        <div id="mdp-feedback-content" class="modal-content">
          <h3>Default</h3>
        </div>
      </div>
      """

    runDemo: () =>
      @feedback = false
      @timeLeft = 1

      actions = OPTIMAL[@trial_i]
      i = 0
      interval = ifvisible.onEvery 1, =>
        if ifvisible.now()
          a = actions[i]
          if a.is_click
            @clickState @states[a.state], a.state
          else
            s = _.last @data.path
            @handleKey s, a.move
          i += 1
          if i is actions.length
            do interval.stop
            # window.clearInterval ID


    # ---------- Responding to user input ---------- #

    # Called when a valid action is initiated via a key press.
    handleKey: (s0, a) =>
      LOG_DEBUG 'handleKey', s0, a
      @data.actions.push a
      @data.actionTimes.push (Date.now() - @initTime)
      unless @disableClicks
        @updatePR TERM_ACTION
        @disableClicks = true
      s1 = @transition[s0][a]
      r = @stateRewards[s1]
      LOG_DEBUG "#{s0}, #{a} -> #{r}, #{s1}"

      s1g = @states[s1]
      @player.animate {left: s1g.left, top: s1g.top},
          duration: MOVE_SPEED
          onChange: @canvas.renderAll.bind(@canvas)
          onComplete: =>
            @addScore r
            if @feedback
              @displayFeedback a, s1
            else
              @arrive s1
            # @arrive s1

    # Called when a state is clicked on.
    clickState: (g, s) =>
      if @disableClicks
        return
      LOG_DEBUG "clickState #{s}"
      if @complete or s is @initial
        return
      if @stateLabels and @stateDisplay is 'click' and not g.label.text
        @addScore -@stateClickCost
        r = @getStateLabel s
        g.setLabel r
        @recordQuery 'click', 'state', s
        delay 0, => @updatePR (parseInt s), r

    updatePR: (action, r) ->
      state = @beliefState.slice()
      @PRdata = @PRdata.then (data) ->
        arg = {state, action}
        callWebppl('getQV', arg).then (qv) ->
          newData = _.extend(qv, arg)
          console.log('PR info', newData)
          data.concat([newData])
      @PRdata.catch (reason) =>
        console.log('WEBPPL ERROR: ' + reason)

      unless action is TERM_ACTION
        @beliefState[action] = r
        @data.beliefs.push @beliefState.slice()

    mouseoverState: (g, s) =>
      LOG_DEBUG "mouseoverState #{s}"
      if @stateLabels and @stateDisplay is 'hover'
        g.setLabel (@getStateLabel s)
        @recordQuery 'mouseover', 'state', s

    mouseoutState: (g, s) =>
      LOG_DEBUG "mouseoutState #{s}"
      if @stateLabels and @stateDisplay is 'hover'
        g.setLabel ''
        @recordQuery 'mouseout', 'state', s

    clickEdge: (g, s0, r, s1) =>
      LOG_DEBUG "clickEdge #{s0} #{r} #{s1}"
      if @edgeLabels and @edgeDisplay is 'click' and not g.label.text
        g.setLabel @getEdgeLabel s0, r, s1
        @recordQuery 'click', 'edge', "#{s0}__#{s1}"

    mouseoverEdge: (g, s0, r, s1) =>
      LOG_DEBUG "mouseoverEdge #{s0} #{r} #{s1}"
      if @edgeLabels and @edgeDisplay is 'hover'
        g.setLabel @getEdgeLabel s0, r, s1
        @recordQuery 'mouseover', 'edge', "#{s0}__#{s1}"

    mouseoutEdge: (g, s0, r, s1) =>
      LOG_DEBUG "mouseoutEdge #{s0} #{r} #{s1}"
      if @edgeLabels and @edgeDisplay is 'hover'
        g.setLabel ''
        @recordQuery 'mouseout', 'edge', "#{s0}__#{s1}"

    getEdgeLabel: (s0, r, s1) =>
      if @edgeLabels is 'reward'
        String(r)
      else
        @edgeLabels["#{s0}__#{s1}"]

    getStateLabel: (s) =>
      if @stateLabels?
        switch @stateLabels
          when 'custom'
            ':)'
          when 'reward'
            @stateRewards[s]
            # '®'
          else
            @stateLabels[s]
      else ''

    # getReward: (s0, a, s1) =>
    #   if @stateRewards
    #     @stateRewards[s1]
    #   else
    #     @graph[s0][a]

    # getOutcome: (s0, a) =>
    #   return 1
    #   [r, s1] = @graph[s0][a]
    #   if @stateRewards
    #     r = @stateRewards[s1]
    #   return [r, s1]

    recordQuery: (queryType, targetType, target) =>
      @canvas.renderAll()
      LOG_DEBUG "recordQuery #{queryType} #{targetType} #{target}"
      # @data["#{queryType}_#{targetType}_#{target}"]
      @data.queries[queryType][targetType].target.push target
      @data.queries[queryType][targetType].time.push Date.now() - @initTime

    displayFeedback: (a, s1) =>
      # @arrive s1
      # return
      @PRdata.then (data) =>
        @arrive s1
      return 

      if not @feedback
        $('#mdp-feedback').css(display: 'none')
        @arrive s1
        return

      
      result.delay = Math.round result.delay  
      console.log 'feedback', result
    
      showCriticism = result.delay>=1
      if PARAMS.PR_type is 'none'
        result.delay = switch PARAMS.info_cost
          when 0.01 then [null, 4, 0, 1][@data.actions.length]
          when 1.00 then [null, 3, 0, 1][@data.actions.length]
          when 2.50 then [null, 15, 0, 3][@data.actions.length]
          when 1.0001 then [null, 2, 0, 1][@data.actions.length]
            
      @data.delays.push result.delay
      @data.planned_too_little.push result.planned_too_little
      @data.planned_too_much.push result.planned_too_much
      @data.information_used_correctly.push result.information_used_correctly

            
      redGreenSpan = (txt, val) ->
        "<span style='color: #{redGreen val}; font-weight: bold;'>#{txt}</span>"


      
      if PARAMS.message
          if PARAMS.PR_type is 'objectLevel'                
                #if the move was optimal, say so
                if a is result.optimal_action.direction
                    head = redGreenSpan "You chose the best possible move.", 1            
                else
                    head = redGreenSpan "Bad move! You should have moved #{result.optimal_action.direction}.", -1            
                
                #if the move was sub-optimal point out the optimal move
          else            
              if PARAMS.message is 'full'
                if result.planned_too_little and showCriticism
                  if result.planned_too_much and showCriticism
                      head = redGreenSpan "You gathered the wrong information.", -1            
                  else
                      head = redGreenSpan "You gathered too little information.", -1            
                else
                  if result.planned_too_much and showCriticism
                      head = redGreenSpan "You gathered too much information.", -1                    
                  else
                      if !result.planned_too_much & !result.planned_too_little        
                        head = redGreenSpan "You gathered the right amount of information.", 1
                      
                      if result.information_used_correctly and showCriticism
                        head += redGreenSpan " But you didn't prioritize the most important locations.", -1
                        
              if PARAMS.message is 'simple'
                    head =''
                    #if result.PR_type is 'none'
                    #    head = ''
                    #else
                    #    head = redGreenSpan "Poor planning!", -1                    
              if PARAMS.message is 'none'
                    if result.delay is 1
                        head = "Please wait 1 second."
                    else
                        head = "Please wait "+result.delay+" seconds."
        
        if PARAMS.PR_type is "none"
            penalty = if result.delay then "<p>Please wait #{result.delay} seconds.</p>"
        else
            penalty = if result.delay then redGreenSpan "<p>#{result.delay} second penalty!</p>", -1
        
        info = do ->
          if PARAMS.message is 'full'
            "Given the information you collected, your decision was " + \
            if result.information_used_correctly
              redGreenSpan 'optimal.', 1
            else
              redGreenSpan 'suboptimal.', -1
          else ''
        
        if (PARAMS.message is 'full' or PARAMS.message is 'simple') and PARAMS.PR_type != 'objectLevel'
            msg = """
            <h3>#{head}</h3>            
            <b>#{penalty}</b>                        
            #{info}
            """
        if PARAMS.PR_type is 'objectLevel'
             msg = """
            <h3>#{head}</h3>             
            <b>#{penalty}</b> 
             """
        
        if PARAMS.message is 'none'
            msg = """
            <h3>#{head}</h3>
            """
      if !PARAMS.message
        msg = "Please wait "+result.delay+" seconds."  

      if @feedback and result.delay>=1        
          @freeze = true
          $('#mdp-feedback').css display: 'block'
          $('#mdp-feedback-content')
            # .css
            #   'background-color': if mistake then RED else GREEN
            #   color: 'white'
            .html msg

          setTimeout (=>
            @freeze = false
            $('#mdp-feedback').css(display: 'none')
            @arrive s1
          ), (if false then 1000 else result.delay * 1000)
      else
        $('#mdp-feedback').css(display: 'none')
        @arrive s1




    # ---------- Updating state ---------- #

    # Called when the player arrives in a new state.
    arrive: (s) =>
      @PRdata = new Promise (resolve) -> resolve([])
      LOG_DEBUG 'arrive', s
      @data.path.push s

      # Get available actions.
      if @transition[s]
        keys = (@keys[a] for a in (Object.keys @transition[s]))
      else
        keys = []
      if not keys.length
        @complete = true
        @checkFinished()
        return

      # Start key listener.
      if not @demonstrate
        @keyListener = jsPsych.pluginAPI.getKeyboardResponse
          valid_responses: keys
          rt_method: 'date'
          persist: false
          allow_held_key: false
          callback_function: (info) =>
            action = @invKeys[info.key]
            LOG_DEBUG 'key', info.key
            @data.rt.push info.rt
            @handleKey s, action

    addScore: (v) =>
      @data.score = round (@data.score + v)
      $('#mouselab-score').html '$' + @data.score.toFixed(2)
      $('#mouselab-score').css 'color', redGreen @data.score


    # ---------- Starting the trial ---------- #

    run: =>
      LOG_DEBUG 'run'
      # meta_MDP.init @trial_i
      do @buildMap
      do @startTimer
      fabric.Image.fromURL @playerImage, ((img) =>
        @initPlayer img
        do @canvas.renderAll
        @initTime = Date.now()
        @arrive @initial
        if @demonstrate
          do @runDemo
      )

    # Draw object on the canvas.
    draw: (obj) =>
      @canvas.add obj
      return obj

    startTimer: =>
      @timeLeft = @minTime
      intervalID = undefined

      interval = ifvisible.onEvery 1, =>
        if @freeze then return
        @timeLeft -= 1
        $('#mdp-time').html @timeLeft
        $('#mdp-time').css 'color', (redGreen (-@timeLeft + .1))  # red if > 0
        if @timeLeft is 0
          do interval.stop
          do @checkFinished
      
      $('#mdp-time').html @timeLeft
      $('#mdp-time').css 'color', (redGreen (-@timeLeft + .1))
      # intervalID = window.setInterval tick, 1000

    # Draws the player image.
    initPlayer: (img) =>
      LOG_DEBUG 'initPlayer'
      top = @states[@initial].top
      left = @states[@initial].left
      img.scale(0.3)
      img.set('top', top).set('left', left)
      @draw img
      @player = img

    # Constructs the visual display.
    buildMap: =>
      do =>
        [xs, ys] = _.unzip (_.values @layout)

        minx = _.min xs
        miny = _.min ys

        xs = xs.map((x) -> x - minx)
        ys = ys.map((y) -> (y - miny))

        @layout = _.zip xs, ys

        width = (_.max xs) + 1
        height = (_.max ys) + 1

        @canvasElement.attr(width: width * SIZE, height: height * SIZE)

      @canvas = new fabric.Canvas 'mouselab-canvas', selection: false

      @states = []
      @beliefState = []
      @layout.forEach (loc, idx) =>
        @beliefState.push UNKNOWN
        [x, y] = loc
        @states.push @draw new State idx, x, y,
          fill: '#bbb'
          # label: if @stateDisplay is 'always' then (@getStateLabel s) else ''
          label: ''
      @beliefState[0] = 0
      @data.beliefs.push @beliefState.slice()

      # LOG_DEBUG '@graph', @graph
      LOG_INFO '@states', @states

      @tree.forEach (s1s, s0) =>
        s1s.forEach (s1) =>
          @draw new Edge @states[s0], 0, @states[s1],
            # label: if @edgeDisplay is 'always' then @getEdgeLabel s0, r, s1 else ''
            label: ''

      # for s0, actions of @graph
      #   for a, [r, s1] of actions
      #     @draw new Edge @states[s0], r, @states[s1],
      #       label: if @edgeDisplay is 'always' then @getEdgeLabel s0, r, s1 else ''



    # ---------- ENDING THE TRIAL ---------- #

    # Creates a button allowing user to move to the next trial.
    endTrial: =>
      # if @demonstrate
        # @lowerMessage.html "<b>Press any key to continue.</b>"
      # else
      SCORE += @data.score
      @lowerMessage.html """
        So far, you've earned a bonus of $#{calculateBonus().toFixed(2)}
        <br>
        <b>Press any key to continue.</b><e
      """
      @keyListener = jsPsych.pluginAPI.getKeyboardResponse
        valid_responses: []
        rt_method: 'date'
        persist: false
        allow_held_key: false
        callback_function: (info) =>
          @display.empty()
          jsPsych.finishTrial @data

    checkFinished: =>
      if @complete and @timeLeft? and @timeLeft > 0
        @lowerMessage.html """Waiting for the timer to expire..."""
      if @complete and @timeLeft <= 0
        do @endTrial


  #  =========================== #
  #  ========= Graphics ========= #
  #  =========================== #

  class State extends fabric.Group
    constructor: (@name, left, top, config={}) ->
      left = (left + 0.5) * SIZE
      top = (top + 0.5) * SIZE
      conf =
        left: left
        top: top
        fill: '#bbbbbb'
        radius: SIZE / 4
        label: ''
      _.extend conf, config

      # Due to a quirk in Fabric, the maximum width of the label
      # is set when the object is initialized (the call to super).
      # Thus, we must initialize the label with a placeholder, then
      # set it to the proper value afterwards.
      @circle = new fabric.Circle conf
      @label = new Text '----------', left, top,
        fontSize: SIZE / 6
        fill: '#44d'

      @radius = @circle.radius
      @left = @circle.left
      @top = @circle.top

      if not mdp.demonstrate
        @on('mousedown', -> mdp.clickState this, @name)
        @on('mouseover', -> mdp.mouseoverState this, @name)
        @on('mouseout', -> mdp.mouseoutState this, @name)
      super [@circle, @label]
      @setLabel conf.label


    setLabel: (txt) ->
      # LOG_DEBUG "setLabel #{txt}"
      if "#{txt}"
        @label.setText "$#{txt}"
        @label.setFill (redGreen txt)
      else
        @label.setText ''
      @dirty = true


  class Edge extends fabric.Group
    constructor: (c1, reward, c2, config={}) ->
      {
        spacing=8
        adjX=0
        adjY=0
        rotateLabel=false
        label=''
      } = config

      [x1, y1, x2, y2] = [c1.left + adjX, c1.top + adjY, c2.left + adjX, c2.top + adjY]

      @arrow = new Arrow(x1, y1, x2, y2,
                         c1.radius + spacing, c2.radius + spacing)

      ang = (@arrow.ang + Math.PI / 2) % (Math.PI * 2)
      if 0.5 * Math.PI <= ang <= 1.5 * Math.PI
        ang += Math.PI
      [labX, labY] = polarMove(x1, y1, angle(x1, y1, x2, y2), SIZE * 0.45)

      # See note about placeholder in State.
      @label = new Text '----------', labX, labY,
        angle: if rotateLabel then (ang * 180 / Math.PI) else 0
        fill: redGreen label
        fontSize: SIZE / 6
        textBackgroundColor: 'white'

      @on('mousedown', -> mdp.clickEdge this, c1.name, reward, c2.name)
      @on('mouseover', -> mdp.mouseoverEdge this, c1.name, reward, c2.name)
      @on('mouseout', -> mdp.mouseoutEdge this, c1.name, reward, c2.name)
      super [@arrow, @label]
      @setLabel label

    setLabel: (txt) ->
      # LOG_DEBUG "setLabel #{txt}"
      if txt
        @label.setText "#{txt}"
        @label.setFill (redGreen txt)
      else
        @label.setText ''
      @dirty = true
      


  class Arrow extends fabric.Group
    constructor: (x1, y1, x2, y2, adj1=0, adj2=0) ->
      @ang = ang = (angle x1, y1, x2, y2)
      [x1, y1] = polarMove(x1, y1, ang, adj1)
      [x2, y2] = polarMove(x2, y2, ang, - (adj2+7.5))

      line = new fabric.Line [x1, y1, x2, y2],
        stroke: '#555'
        selectable: false
        strokeWidth: 3

      @centerX = (x1 + x2) / 2
      @centerY = (y1 + y2) / 2
      deltaX = line.left - @centerX
      deltaY = line.top - @centerY
      dx = x2 - x1
      dy = y2 - y1

      point = new (fabric.Triangle)(
        left: x2 + deltaX
        top: y2 + deltaY
        pointType: 'arrow_start'
        angle: ang * 180 / Math.PI
        width: 10
        height: 10
        fill: '#555')

      super [line, point]


  class Text extends fabric.Text
    constructor: (txt, left, top, config) ->
      txt = String(txt)
      conf =
        left: left
        top: top
        fontFamily: 'helvetica'
        fontSize: SIZE / 8

      _.extend conf, config
      super txt, conf


  # ================================= #
  # ========= jsPsych stuff ========= #
  # ================================= #
  
  plugin =
    trial: (display_element, trialConfig) ->
      trialConfig = jsPsych.pluginAPI.evaluateFunctionParameters(trialConfig)
      trialConfig.display = display_element

      console.log 'trialConfig', trialConfig

      display_element.empty()
      trial = new MouselabMDP trialConfig
      trial.run()
      if trialConfig._block
        trialConfig._block.trialCount += 1
      TRIAL_INDEX += 1

  return plugin

# ---
# generated by js2coffee 2.2.0