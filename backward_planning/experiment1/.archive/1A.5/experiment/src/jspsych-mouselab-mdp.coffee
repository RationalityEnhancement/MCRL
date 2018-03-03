###
jspsych-mouselab-mdp.coffee
Fred Callaway

https://github.com/fredcallaway/Mouselab-MDP
###

# coffeelint: disable=max_line_length
mdp = undefined
D = undefined
WATCH = {}
SCORE = 0
TIME_LEFT = undefined

jsPsych.plugins['mouselab-mdp'] = do ->


  PRINT = (args...) -> console.log args...
  NULL = (args...) -> null
  LOG_INFO = PRINT
  LOG_DEBUG = NULL

  # a scaling parameter, determines size of drawn objects
  SIZE = undefined
  TRIAL_INDEX = 0
  TOP_ADJUST = -16

  fabric.Object::originX = fabric.Object::originY = 'center'
  fabric.Object::selectable = false
  fabric.Object::hoverCursor = 'plain'

  # =========================== #
  # ========= Helpers ========= #
  # =========================== #

  removePrivate = (obj) ->
    _.pick obj, ((v, k, o) -> not k.startsWith('_'))

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
      '#666'

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

  KEYS = mapObject
    up: 'uparrow'
    down: 'downarrow',
    right: 'rightarrow',
    left: 'leftarrow',
    simulate: 'space'
    jsPsych.pluginAPI.convertKeyCharacterToKeyCode
  
  RIGHT_MESSAGE = '\xa0'.repeat(8) + 'Score: <span id=mouselab-score/>'

# =============================== #
# ========= MouselabMDP ========= #
# =============================== #
  
  class MouselabMDP
    constructor: (config) ->
      {
        @display  # html display element
        
        @graph  # defines transition and reward functions
        @layout  # defines position of states
        @initial  # initial state of player

        @stateLabels='reward'  # object mapping from state names to labels
        @stateDisplay='never'  # one of 'never', 'hover', 'click', 'always'
        @stateClickCost=0  # subtracted from score every time a state is clicked
        @edgeLabels='never'  # object mapping from edge names (s0 + '__' + s1) to labels
        @edgeDisplay='always'  # one of 'never', 'hover', 'click', 'always'
        @edgeClickCost=0  # subtracted from score every time an edge is clicked
        @stateRewards=null

        @clickDelay=0
        @moveDelay=500
        @clickEnergy=0
        @moveEnergy=0
        @startScore=0

        @actions=null
        @clicks=null
        @pid=null

        @allowSimulation=false
        @revealRewards=true
        @training=false
        @special=''
        @timeLimit=null
        @minTime=null
        @energyLimit=null
        @qs=null

        # @transition=null  # function `(s0, a, s1, r) -> null` called after each transition
        @keys=KEYS  # mapping from actions to keycodes
        @trialIndex=TRIAL_INDEX  # number of trial (starts from 1)
        @playerImage='static/images/plane.png'
        size=80  # determines the size of states, text, etc...

        # leftMessage="Round: #{TRIAL_INDEX}/#{N_TRIAL}"
        trial_id=null
        blockName='none'
        prompt='&nbsp;'
        leftMessage='&nbsp;'
        centerMessage='&nbsp;'
        rightMessage=RIGHT_MESSAGE
        lowerMessage='&nbsp;'
      } = config

      @termAction = "#{@stateRewards.length}"
      if @pid?
        @showParticipant = true
        centerMessage = "Participant #{@pid}"

      SIZE = size

      _.extend this, config
      checkObj this

      if @stateLabels is 'reward'
        @stateLabels = @stateRewards
      @stateLabels[0] = ''

      if @energyLimit
        leftMessage = 'Energy: <b><span id=mouselab-energy/></b>'
        if not @_block.energyLeft?
          @_block.energyLeft = @energyLimit
      else
        leftMessage = "Round #{@_block.trialCount + 1}/#{@_block.timeline.length}"
        # leftMessage = "Round #{@_block.trialCount + 1}/#{@_block.timeline.length}"

      @data =
        stateRewards: @stateRewards
        trial_id: trial_id
        block: blockName
        trialIndex: @trialIndex
        score: 0
        simulationMode: []
        rewards: []
        path: []
        rt: []
        actions: []
        actionTimes: []
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

      if $('#mouselab-msg-right').length  # right message already exists
        @leftMessage = $('#mouselab-msg-left')
        @leftMessage.html leftMessage
        @centerMessage = $('#mouselab-msg-center')
        @centerMessage.html centerMessage
        @rightMessage = $('#mouselab-msg-right')
        @rightMessage.html rightMessage
        @stage = $('#mouselab-stage')
        @prompt = $('#mouselab-prompt')
        @prompt.html prompt
        # @canvasElement = $('#mouselab-canvas')
        # @lowerMessage = $('#mouselab-msg-bottom')
      else
        do @display.empty
        # @display.css 'width', '1000px'

        # leftMessage = "Round: #{@trialIndex + 1}/#{@_block.timeline.length}"
        @leftMessage = $('<div>',
          id: 'mouselab-msg-left'
          class: 'mouselab-header'
          html: leftMessage).appendTo @display

        @centerMessage = $('<div>',
          id: 'mouselab-msg-center'
          class: 'mouselab-header'
          html: centerMessage).appendTo @display

        @rightMessage = $('<div>',
          id: 'mouselab-msg-right',
          class: 'mouselab-header'
          html: rightMessage).appendTo @display

        unless prompt is null
          @prompt = $('<div>',
            id: 'mouselab-prompt'
            class: 'mouselab-prompt'
            html: prompt).appendTo @display

        @stage = $('<div>',
          id: 'mouselab-stage').appendTo @display

        if @timeLimit
          TIME_LEFT = @timeLimit

        @addScore @startScore
      # -----------------------------

      @canvasElement = $('<canvas>',
        id: 'mouselab-canvas',
      ).attr(width: 500, height: 500).appendTo @stage

      @lowerMessage = $('<div>',
        id: 'mouselab-msg-bottom'
        class: 'mouselab-msg-bottom'
        html: lowerMessage or '&nbsp'
      ).appendTo @stage

      @waitMessage = $('<div>',
        id: 'mouselab-wait-msg'
        class: 'mouselab-msg-bottom'
        # html: """Please wait <span id='mdp-time'></span> seconds"""
      ).appendTo @display

      @waitMessage.hide()
      @defaultLowerMessage = lowerMessage

      # feedback element
      $('#jspsych-target').append """
      <div id="mdp-feedback" class="modal">
        <div id="mdp-feedback-content" class="modal-content">
          <h3>Default</h3>
        </div>
      </div>
      """

      mdp = this
      LOG_INFO 'new MouselabMDP', this
      @invKeys = _.invert @keys
      @resetScore()
      @spendEnergy 0
      @freeze = false
      @lowerMessage.css 'color', '#000'

    runDemo: () =>
      @timeLeft = 1
      console.log 'runDemo'
      for c in @clicks
        await sleep 1000
        console.log 'click', c
        @clickState @states[c], c
        @canvas.renderAll()
      
      for a in @actions
        await sleep 700
        s = _.last @data.path
        @handleKey s, a
          

    startTimer: =>
      @timeLeft = @minTime
      @waitMessage.html "Please wait #{@timeLeft} seconds"
      interval = ifvisible.onEvery 1, =>
        if @freeze then return
        @timeLeft -= 1
        @waitMessage.html "Please wait #{@timeLeft} seconds"
        # $('#mdp-time').html @timeLeft
        # $('#mdp-time').css 'color', (redGreen (-@timeLeft + .1))  # red if > 0
        if @timeLeft is 0
          do interval.stop
          do @checkFinished
      
      $('#mdp-time').html @timeLeft
      $('#mdp-time').css 'color', (redGreen (-@timeLeft + .1))

    endBlock: () ->
      @blockOver = true
      jsPsych.pluginAPI.cancelAllKeyboardResponses()
      @keyListener = jsPsych.pluginAPI.getKeyboardResponse
        valid_responses: ['space']
        rt_method: 'date'
        persist: false
        allow_held_key: false
        callback_function: (info) =>
          jsPsych.finishTrial @data
          do @display.empty
          do jsPsych.endCurrentTimeline

    # ---------- Responding to user input ---------- #

    # Called when a valid action is initiated via a key press.
    handleKey: (s0, a) =>
      LOG_DEBUG 'handleKey', s0, a
      if a is 'simulate'
        if @simulationMode
          @endSimulationMode()
        else
          @startSimulationMode()
      else
        if not @simulationMode
          @allowSimulation = false
          if @defaultLowerMessage
            @lowerMessage.html 'Move with the arrow keys.'
            @lowerMessage.css 'color', '#000'


        @data.actions.push a
        @data.simulationMode.push @simulationMode
        @data.actionTimes.push (Date.now() - @initTime)

        [_, s1] = @graph[s0][a]
        # LOG_DEBUG "#{s0}, #{a} -> #{r}, #{s1}"
        @move s0, a, s1

    startSimulationMode: () =>
      @simulationMode = true
      @player.set('top', @states[@initial].top - 20).set('left', @states[@initial].left)
      @player.set('opacity', 0.4)
      @canvas.renderAll()
      @arrive @initial
      # @centerMessage.html 'Ghost Score: <span id=mouselab-ghost-score/>'
      @rightMessage.html 'Ghost Score: <span id=mouselab-score/>'
      @resetScore()
      @drawScore @data.score
      @lowerMessage.html """
      <b>👻 Ghost Mode 👻</b>
      <br>
      Press <code>space</code> to return to your corporeal form.
      """
    
    endSimulationMode: () =>
      @simulationMode = false
      @player.set('top', @states[@initial].top).set('left', @states[@initial].left)
      @player.set('opacity', 1)
      @canvas.renderAll()
      @arrive @initial
      @centerMessage.html ''
      @rightMessage.html RIGHT_MESSAGE
      @resetScore()
      @lowerMessage.html @defaultLowerMessage

    getOutcome: (s0, a) =>
      LOG_DEBUG "getOutcome #{s0}, #{a}"
      [s1, r] = @graph[s0][a]
      if @stateRewards?
        r = @stateRewards[s1]
      return [r, s1]

    encodeBelief: =>
      b = _.values(@states)
        .map((g) => g.label.text or '_')
      b[0] = 0  # first state is known to be 0
      return b.join(' ')

    getReward: (s0, a, s1) =>
      if @stateRewards?
        @stateRewards[s1]
      else
        @graph[s0][a]

    move: (s0, a, s1) =>
      unless @moved
        await @showFeedback @termAction
      @moved = true
      if @freeze
        LOG_INFO 'freeze!'
        @arrive s0, 'repeat'
        return

      nClick = @data.queries.click.state.target.length
      notEnoughClicks = (@special.startsWith 'trainClick') and nClick < 3
      if notEnoughClicks
        @lowerMessage.html '<b>Inspect at least three nodes before moving!</b>'
        @lowerMessage.css 'color', '#FC4754'
        @special = 'trainClickBlock'
        @arrive s0, 'repeat'
        return

      r = @getReward s0, a, s1
      LOG_DEBUG "move #{s0}, #{s1}, #{r}"
      s1g = @states[s1]
      @freeze = true
      
      newTop = if @simulationMode then s1g.top - 20 else s1g.top + TOP_ADJUST
      @player.animate {left: s1g.left, top: newTop},
        duration: @moveDelay
        onChange: @canvas.renderAll.bind(@canvas)
        onComplete: =>
          @data.rewards.push r
          @addScore r
          @spendEnergy @moveEnergy
          @arrive s1

    clickState: (g, s) =>
      LOG_INFO "clickState #{s}"
      if @moved
        @lowerMessage.html "<b>You can't use the node inspector after moving!</b>"
        @lowerMessage.css 'color', '#FC4754'
        return
              
      if @complete or ("#{s}" is "#{@initial}") or @freeze
        return

      if @special is 'trainClickBlock' and @data.queries.click.state.target.length == 2
        @lowerMessage.html '<b>Nice job! You can click on more nodes or start moving.</b>'
        @lowerMessage.css 'color', '#000'


      if @stateLabels and @stateDisplay is 'click' and not g.label.text
        await @showFeedback s # Note: this must be called before g.setLabel r
        @addScore -@stateClickCost
        @recordQuery 'click', 'state', s
        @spendEnergy @clickEnergy
        r = @stateLabels[s]
        if @clickDelay
          @freeze = true
          g.setLabel '...'
          await sleep @clickDelay()
          @freeze = false
        g.setLabel r
        @canvas.renderAll()

    showFeedback: (action) =>
              
      console.log 'showFeedback'
      qs = @qs[@encodeBelief()]
      v = (_.max qs)
      optimal = (a for a, q of qs when v - q < .01)

      if action in optimal
        return

      @freeze = true
      strictness = 1
      loss = v - qs[action]
      if loss > 0
        delay = 2 + Math.round(strictness * loss)
      else
        delay = 0
      
    
      if @_block.show_feedback
          oldFeedbackMessage = @prompt.html()

          if @termAction in optimal
            msg = """        
              You shouldn't have inspected any more nodes.
            """
          else
            msg = """          
              You should have inspected one of the highlighted nodes.          
            """
            for a in optimal
              @states[a].circle.set('fill', '#49f')
            @canvas.renderAll()

          @prompt.html """
            <div align='center' style='color:#FF0000; font-weight:bold; font-size:18pt'>
            #{msg}<br>
            Please wait #{delay} seconds.
            </div>
          """


          # @freeze = true
          # $('#mdp-feedback').show()
          # $('#mdp-feedback-content')
          #   .html msg
          # $('#mdp-feedback').hide()

          await sleep delay * 1000

          # Reset.
          @prompt.html oldFeedbackMessage
      else
            console.log 'no'

             
      @freeze = false
      unless @termAction in optimal
        for s in optimal
          @states[s].circle.set('fill', '#bbb')
        @canvas.renderAll()
 
            
    mouseoverState: (g, s) =>
      # LOG_DEBUG "mouseoverState #{s}"
      if @stateLabels and @stateDisplay is 'hover'
        # webppl.run('flip()', (s, x) -> g.setLabel (Number x))
        g.setLabel @stateLabels[s]
        @recordQuery 'mouseover', 'state', s

    mouseoutState: (g, s) =>
      # LOG_DEBUG "mouseoutState #{s}"
      if @stateLabels and @stateDisplay is 'hover'
        g.setLabel ''
        @recordQuery 'mouseout', 'state', s

    clickEdge: (g, s0, r, s1) =>
      if not @complete and g.label.text is '?'
        LOG_DEBUG "clickEdge #{s0} #{r} #{s1}"
        if @edgeLabels and @edgeDisplay is 'click' and g.label.text in ['?', '']
          g.setLabel @getEdgeLabel s0, r, s1
          @recordQuery 'click', 'edge', "#{s0}__#{s1}"

    mouseoverEdge: (g, s0, r, s1) =>
      # LOG_DEBUG "mouseoverEdge #{s0} #{r} #{s1}"
      if @edgeLabels and @edgeDisplay is 'hover'
        g.setLabel @getEdgeLabel s0, r, s1
        @recordQuery 'mouseover', 'edge', "#{s0}__#{s1}"

    mouseoutEdge: (g, s0, r, s1) =>
      # LOG_DEBUG "mouseoutEdge #{s0} #{r} #{s1}"
      if @edgeLabels and @edgeDisplay is 'hover'
        g.setLabel ''
        @recordQuery 'mouseout', 'edge', "#{s0}__#{s1}"

    getEdgeLabel: (s0, r, s1) =>
      if @edgeLabels is 'reward'
        '®'
      else
        @edgeLabels["#{s0}__#{s1}"]

    recordQuery: (queryType, targetType, target) =>
      @canvas.renderAll()
      # LOG_DEBUG "recordQuery #{queryType} #{targetType} #{target}"
      # @data["#{queryType}_#{targetType}_#{target}"]
      @data.queries[queryType][targetType].target.push target
      @data.queries[queryType][targetType].time.push Date.now() - @initTime


    # ---------- Updating state ---------- #

    # Called when the player arrives in a new state.
    arrive: (s, repeat=false) =>
      g = @states[s]
      g.setLabel @stateRewards[s]
      @canvas.renderAll()
      @freeze = false
      LOG_DEBUG 'arrive', s

      unless repeat  # sending back to previous state
        @data.path.push s

      # Get available actions.
      if @graph[s]
        keys = (@keys[a] for a in (Object.keys @graph[s]))
      else
        keys = []
      if @allowSimulation
        keys.push 'space'
      if not keys.length
        @complete = true
        @checkFinished()
        return

      unless mdp.showParticipant
        # Start key listener.
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
      @data.score += v
      if @simulationMode
        score = @data.score
      else
        SCORE += v
        score = SCORE
      @drawScore(score)

    resetScore: =>
      @data.score = 0
      @drawScore SCORE

    drawScore: (score)=>
      $('#mouselab-score').html ('$' + score)
      $('#mouselab-score').css 'color', redGreen score

    spendEnergy: (v) =>
      @_block.energyLeft -= v
      if @_block.energyLeft <= 0
        LOG_INFO 'OUT OF ENERGY'
        @_block.energyLeft = 0
        @freeze = true
        @lowerMessage.html """<b>You're out of energy! Press</b> <code>space</code> <b>to continue.</br>"""
        @endBlock()
      $('#mouselab-energy').html @_block.energyLeft
      # $('#mouselab-').css 'color', redGreen SCORE

          


    # ---------- Starting the trial ---------- #

    run: =>
      # document.body.style.cursor = 'crosshair'
      jsPsych.pluginAPI.cancelAllKeyboardResponses()
      LOG_DEBUG 'run'
      @buildMap()
      if @timeLimit or @minTime
        do @startTimer
      fabric.Image.fromURL @playerImage, ((img) =>
        @initPlayer img
        @canvas.renderAll()
        @initTime = Date.now()
        @arrive @initial
      )
      if @showParticipant
        @runDemo()
    # Draw object on the canvas.
    draw: (obj) =>
      @canvas.add obj
      return obj


    # Draws the player image.
    initPlayer: (img) =>
      LOG_DEBUG 'initPlayer'
      top = @states[@initial].top + TOP_ADJUST
      left = @states[@initial].left
      img.scale(0.25)
      img.set('top', top).set('left', left)
      @draw img
      @player = img

    # Constructs the visual display.
    buildMap: =>
      # Resize canvas.
      [xs, ys] = _.unzip (_.values @layout)
      minx = _.min xs
      miny = _.min ys
      maxx = _.max xs
      maxy = _.max ys
      [width, height] = [maxx - minx + 1, maxy - miny + 1]

      @canvasElement.attr(width: width * SIZE, height: height * SIZE)
      @canvas = new fabric.Canvas 'mouselab-canvas', selection: false
      @canvas.defaultCursor = 'pointer'

      @states = {}
      for s, location of (removePrivate @layout)
        [x, y] = location

        @states[s] = new State s, x - minx, y - miny,
          fill: '#bbb'
          label: if @stateDisplay is 'always' then @stateLabels[s] else ''

      for s0, actions of (removePrivate @graph)
        for a, [r, s1] of actions
          new Edge @states[s0], r, @states[s1],
            label: if @edgeDisplay is 'always' then @getEdgeLabel s0, r, s1 else ''


    # ---------- ENDING THE TRIAL ---------- #

    # Creates a button allowing user to move to the next trial.
    endTrial: =>
      window.clearInterval @timerID
      if @blockOver
        return
      @lowerMessage.html """
        You made <span class=mouselab-score/> on this round.
        <br>
        <b>Press</b> <code>space</code> <b>to continue.</b>
      """
      $('.mouselab-score').html '$' + @data.score
      $('.mouselab-score').css 'color', redGreen @data.score
      $('.mouselab-score').css 'font-weight', 'bold'
      @keyListener = jsPsych.pluginAPI.getKeyboardResponse
        valid_responses: ['space']
        rt_method: 'date'
        persist: false
        allow_held_key: false
        callback_function: (info) =>
          @data.trialTime = getTime() - @initTime
          jsPsych.finishTrial @data
          do @stage.empty

    checkFinished: =>
      if @complete
        if @timeLeft?
          if @timeLeft > 0
            @waitMessage.show()
          else
            @waitMessage.hide()
            do @endTrial
        else
          do @endTrial


  #  =========================== #
  #  ========= Graphics ========= #
  #  =========================== #

  class State
    constructor: (@name, left, top, config={}) ->
      left = (left + 0.5) * SIZE
      top = (top + 0.5) * SIZE
      @left = left
      @top = top
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
      @label = {}
      # @label = new Text '----------', left, top,
      #   fontSize: SIZE / 4
      #   fill: '#44d'

      @radius = @circle.radius
      @left = @circle.left
      @top = @circle.top

      mdp.canvas.add(@circle)
      
      # @setLabel conf.label
      unless mdp.showParticipant
        @circle.on('mousedown', => mdp.clickState this, @name)
        @circle.on('mouseover', => mdp.mouseoverState this, @name)
        @circle.on('mouseout', => mdp.mouseoutState this, @name)

    setLabel: (txt, conf={}) ->
      @label = new Text '----------', @left, @top,
        fontSize: SIZE / 4
        fill: '#44d'
      mdp.canvas.add(@label)
      LOG_DEBUG 'setLabel', txt
      {
        pre=''
        post=''
      } = conf
      # LOG_DEBUG "setLabel #{txt}"
      if txt?
        @label.setText "#{pre}#{txt}#{post}"
        @label.setFill (redGreen txt)
      else
        @label.setText ''
      @dirty = true

    higlight: ->
      @circle.set('color', '#49f')




  class Edge
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
        fontSize: SIZE / 4
        textBackgroundColor: 'white'

      @arrow.on('mousedown', => mdp.clickEdge this, c1.name, reward, c2.name)
      @arrow.on('mouseover', => mdp.mouseoverEdge this, c1.name, reward, c2.name)
      @arrow.on('mouseout', => mdp.mouseoutEdge this, c1.name, reward, c2.name)
      @setLabel label

      mdp.canvas.add(@arrow)
      mdp.canvas.add(@label)

    setLabel: (txt, conf={}) ->
      {
        pre=''
        post=''
      } = conf
      # LOG_DEBUG "setLabel #{txt}"
      if txt
        @label.setText "#{pre}#{txt}#{post}"
        @label.setFill (redGreen txt)
      else
        @label.setText ''
      @dirty = true
      


  class Arrow extends fabric.Group
    constructor: (x1, y1, x2, y2, adj1=0, adj2=0) ->
      ang = (angle x1, y1, x2, y2)
      [x1, y1] = polarMove(x1, y1, ang, adj1)
      [x2, y2] = polarMove(x2, y2, ang, - (adj2+7.5))

      line = new fabric.Line [x1, y1, x2, y2],
        stroke: '#555'
        selectable: false
        strokeWidth: 3

      centerX = (x1 + x2) / 2
      centerY = (y1 + y2) / 2
      deltaX = line.left - centerX
      deltaY = line.top - centerY
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
      @ang = ang
      @centerX = centerX
      @centerY = centerY



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
      # trialConfig = jsPsych.pluginAPI.evaluateFunctionParameters trialConfig, ['_init', 'constructor']
      trialConfig.display = display_element

      LOG_INFO 'trialConfig', trialConfig

      trial = new MouselabMDP trialConfig
      trial.run()
      if trialConfig._block
        trialConfig._block.trialCount += 1
      TRIAL_INDEX += 1

  return plugin

# ---
# generated by js2coffee 2.2.0