###
jspsych-plane.coffee
Fred Callaway

An MDP game in which the participant plans flights to
maximize profit.

###

# coffeelint: disable=max_line_length
locations = undefined
canvas = undefined
# the single instance of PlaneGame
game = undefined
jsPsych.plugins['plane'] = do ->

  DEBUG = console.log
  # DEBUG = (args...) -> null

  # ==== GLOBALS ==== #
  # a scaling parameter, determines size of drawn objects
  s = undefined
  # the fabric.Canvas object
  canvas = undefined

  fabric.Object::originX = fabric.Object::originY = 'center'
  fabric.Object::selectable = false
  fabric.Object::hoverCursor = 'plain'


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

  # Draw object on canvas.
  add = (obj) ->
    canvas.add obj
    return obj

  dist = (o1, o2) ->
    ((o1.left - o2.left) ** 2 + (o1.top - o2.top)**2) ** 0.5


  # ======================== #
  # ========= Game ========= #
  # ======================== #
  
  class PlaneGame
    constructor: (config) ->
      {
        @display
        @initial
        @graph
        @final=[]
        @pseudo=null
        @depth=null
        @width=null
        @layout='ring'
        @n_moves=null
        @pTerm=0
        lowerMessage='Control the plane by clicking on the next destination.'
        stim
        pr_depth
      } = config

      
      @data =
        path: [@initial]
      @score = 0
      @pseudoScore = 0
      @noInput = false
      @movesLeft = @n_moves

      @progressCounter = $('<div>',
        id: 'plane-progress-counter'
        class: 'plane-header'
        html: '&nbsp').appendTo @display

      # @message = $('<div>',
      #   id: 'plane-message'
      #   class: 'plane-header'
      #   html: @prompt or '&nbsp').appendTo @display

      @moveCounter = $('<div>',
        id: 'plane-move-counter',
        class: 'plane-header'
        html: 'Moves: <span id=plane-moves/>').appendTo @display

      if not @movesLeft?
        @moveCounter.html '&nbsp'

      pseudo_html = if @pseudo? then 'Stars: <span id=plane-pseudo/>' else '&nbsp'
      @pseudoCounter = $('<div>',
        id: 'plane-pseudo-counter',
        class: 'plane-header'
        html: pseudo_html).appendTo @display

      @scoreCounter = $('<div>',
        id: 'plane-score-counter',
        class: 'plane-header'
        html: 'Profit: <span id=plane-score/>').appendTo @display

      switch @layout
        when 'ring'
          s = 100
          @width = 5.5
          @height = 4
          width = @width * s
          height = @height * s
        when 'pachinko'
          s = 140
          width = (@depth) * s
          height = (@width + .5) * s
          

      @canvas = $('<canvas>',
        id: 'plane-canvas',
      ).attr(width: width, height: height).appendTo(@display)

      @lowerMessage = $('<div>',
        id: 'plane-lower-message'
        html: lowerMessage or '&nbsp').appendTo @display

      @updateCounters()
      
      # Update global vars
      canvas = new fabric.Canvas 'plane-canvas', selection: false
      game = this

      @data = {
        @width
        @height
        stim
        pr_depth
        path: []
      }

      checkObj this
      console.log 'new PlaneGame', this

    updateCounters: =>
      @progressCounter.html "Trial: 1 / 1"
      $('#plane-score').html '$' + @score
      $('#plane-pseudo').html '⭐' + @pseudoScore
      color = if @score is 0 then 'gray' else if @score < 0 then 'red' else 'green'
      $('#plane-score').css 'color', color
      $('#plane-moves').html @movesLeft

    updatePseudo: ->
      if not @pseudo?
        return
      if @nextPseudo
        pr = @nextPseudo[@plane.location]
        if pr?
          @pseudoScore += pr
        else
          return  # haven't reached the next pseudo-reward
      # Compute new pseudo-rewards.
      @nextPseudo = {}
      pseudo = @pseudo[@plane.location]
      for loc in @locations
        pr = pseudo[loc.name]
        if pr?
          @nextPseudo[loc.name] = pr
        else
          pr = ''
        loc.setLabel String(pr)

      canvas.renderAll()

    move: (loc) =>
      if @noInput then return
      source = @plane.location
      connections = @graph[source][1]

      for [dest, r] in connections
        if @plane.location is source and loc.name is dest
          # legal move -> execute it
          @data.path.push dest
          @noInput = true
          @movesLeft -= 1
          @plane.location = loc.name
          @score += r
          @plane.animate {left: loc.left, top: loc.top},
              duration: dist(@plane, loc) * 4
              onChange: canvas.renderAll.bind(canvas)
              onComplete: =>
                @updatePseudo()
                @updateCounters()
                @noInput = false
                if Math.random() < @pTerm or @movesLeft == 0 or dest in @final
                  @endTrial()

    
    endTrial: =>
      console.log 'endTrial'
      @data.score = @score
      @lowerMessage.html """Well done! Click Continue to move on.<br>"""

      $('<button>')
        .addClass('btn btn-primary btn-lg')
        .text('Continue')
        .click (=>
          @display.empty()
          jsPsych.finishTrial @data)
        .appendTo @lowerMessage

    run: =>
      @buildMap()
      fabric.Image.fromURL '/static/images/plane.png', ((img) =>
        top = @locations[@initial].top
        left = @locations[@initial].left
        img.scale(0.35)
        # img.set('top', 0).set('left', 0)  # start at state 0
        img.set('top', top).set('left', left)
        add img
        img.set('top', top).set('left', left)
        canvas.renderAll()
        @plane = img
        @plane.location = @initial
        @updatePseudo()
        # setTimeout
      )

    buildMap: =>
      switch @layout
        when 'ring'
          @locations = [
            add new Location 0, 1, @height / 2
            add new Location 1, 2, @height / 2 - 1
            add new Location 2, 3.5, @height / 2 - 1
            add new Location 3, 4.5, @height / 2
            add new Location 4, 3.5, @height / 2 + 1
            add new Location 5, 2, @height / 2 + 1
          ]
        when 'pachinko'
          @locations = []
          for d in [0...@depth]
            for w in [0...@width]
              adj = if d % 2 then 0.5 else 0
              @locations.push (add new Location d * @width + w, 0.5 + d, 1 + w - adj)

        else
          throw new Error "Bad layout #{@layout}"

      console.log '@locations', @locations
      for [source, connections] in @graph
        for [target, reward] in connections
          add new Connection @locations[source], @locations[target],
            reward: reward
    

  #  =========================== #
  #  ========= Objects ========= #
  #  =========================== #

  # A node in the graph, a city the plane can be in.
  class Location extends fabric.Group
    constructor: (@name, left, top, config={}) ->
      left *= s
      top *= s
      conf =
        left: left
        top: top
        fill: '#bbbbbb'
        radius: s / 6
        hoverCursor: 'pointer'
        label: ''
      _.extend conf, config
      # @x = @left = left
      # @y = @top = top
      @on('mousedown', -> game.move this)
      @circle = new fabric.Circle conf
      @label = new Text conf.label, left, top,
        fontSize: 20
        fill: '#44d'
      @radius = @circle.radius
      @left = @circle.left
      @top = @circle.top
      super [@circle, @label]

    setLabel: (txt) ->
      @label.setText txt
      @dirty = true

  # An edge in the graph, a flight route.
  class Connection extends fabric.Group
    constructor: (c1, c2, conf={}) ->
      {
        reward
        pseudo=null
        label2=''
        spacing=8
        adjX=0
        adjY=0
      } = conf

      [x1, y1, x2, y2] = [c1.left + adjX, c1.top + adjY, c2.left + adjX, c2.top + adjY]

      @arrow = new Arrow(x1, y1, x2, y2,
                     c1.radius + spacing, c2.radius + spacing)

      ang = (@arrow.ang + Math.PI / 2) % (Math.PI * 2)
      if 0.5 * Math.PI <= ang <= 1.5 * Math.PI
        ang += Math.PI
      
      # [labX, labY] = [x1 * 0.65 + x2 * 0.35,
      #                 y1 * 0.65 + y2 * 0.35]

      [labX, labY] = polarMove(x1, y1, angle(x1, y1, x2, y2), s*0.5)

      fill = if reward > 0
        '#080'
      else if reward < 0
        '#b00'
      else if reward == 0
        '#333'

      txt = "$#{reward}" + (if pseudo? then " + #{pseudo}⭐️" else '')
      @label = new Text txt, labX, labY,
        angle: (ang * 180 / Math.PI)
        fill: fill
        fontSize: 20
        textBackgroundColor: 'white'

      # if label2
      #   [labX, labY] = polarMove(labX, labY, ang, -20)
      #   lab = new Text label2, labX, labY,
      #     angle: (ang * 180 / Math.PI)
      #     fill: '#f88'

      super [@arrow, @label]


  class Arrow extends fabric.Group
    constructor: (x1, y1, x2, y2, adj1=0, adj2=0) ->
      @ang = ang = (angle x1, y1, x2, y2)
      [x1, y1] = polarMove(x1, y1, ang, adj1)
      [x2, y2] = polarMove(x2, y2, ang, - (adj2+7.5))

      line = new fabric.Line [x1, y1, x2, y2],
        stroke: '#000'
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
        fill: '#000')

      super [line, point]


  class Text extends fabric.Text
    constructor: (txt, left, top, config) ->
      txt = String(txt)
      conf =
        left: left
        top: top
        fontFamily: 'helvetica'
        fontSize: 14

      _.extend conf, config
      super txt, conf


  # ================================= #
  # ========= jsPsych stuff ========= #
  # ================================= #
  
  plugin = {}
  plugin.trial = (display_element, trial_config) ->
    display_element.empty()
    trial_config = jsPsych.pluginAPI.evaluateFunctionParameters(trial_config)
    trial_config['display'] = display_element

    _.extend trial_config, (loadJson trial_config.stim)
    console.log 'trial_config', trial_config
    trial = new PlaneGame trial_config
    window.trial = trial
    trial.run()

  plugin

# ---
# generated by js2coffee 2.2.0