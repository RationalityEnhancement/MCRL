// Generated by CoffeeScript 1.12.3

/*
jspsych-mouselab-mdp.coffee
Fred Callaway

https://github.com/fredcallaway/Mouselab-MDP
 */
var mdp,
  slice = [].slice,
  bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; },
  extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
  hasProp = {}.hasOwnProperty;

mdp = void 0;

jsPsych.plugins['mouselab-mdp'] = (function() {
  var Arrow, Edge, KEYS, KEY_DESCRIPTION, LOG_DEBUG, LOG_INFO, MouselabMDP, NULL, PRINT, SIZE, State, TRIAL_INDEX, Text, angle, checkObj, dist, plugin, polarMove, redGreen, round;
  PRINT = function() {
    var args;
    args = 1 <= arguments.length ? slice.call(arguments, 0) : [];
    return console.log.apply(console, args);
  };
  NULL = function() {
    var args;
    args = 1 <= arguments.length ? slice.call(arguments, 0) : [];
    return null;
  };
  LOG_INFO = PRINT;
  LOG_DEBUG = NULL;
  SIZE = void 0;
  TRIAL_INDEX = 1;
  fabric.Object.prototype.originX = fabric.Object.prototype.originY = 'center';
  fabric.Object.prototype.selectable = false;
  fabric.Object.prototype.hoverCursor = 'plain';
  angle = function(x1, y1, x2, y2) {
    var ang, x, y;
    x = x2 - x1;
    y = y2 - y1;
    if (x === 0) {
      ang = y === 0 ? 0 : y > 0 ? Math.PI / 2 : Math.PI * 3 / 2;
    } else if (y === 0) {
      ang = x > 0 ? 0 : Math.PI;
    } else {
      ang = x < 0 ? Math.atan(y / x) + Math.PI : y < 0 ? Math.atan(y / x) + 2 * Math.PI : Math.atan(y / x);
    }
    return ang + Math.PI / 2;
  };
  polarMove = function(x, y, ang, dist) {
    x += dist * Math.sin(ang);
    y -= dist * Math.cos(ang);
    return [x, y];
  };
  dist = function(o1, o2) {
    return Math.pow(Math.pow(o1.left - o2.left, 2) + Math.pow(o1.top - o2.top, 2), 0.5);
  };
  redGreen = function(val) {
    if (val > 0) {
      return '#080';
    } else if (val < 0) {
      return '#b00';
    } else {
      return '#888';
    }
  };
  round = function(x) {
    return (Math.round(x * 100)) / 100;
  };
  checkObj = function(obj, keys) {
    var i, k, len;
    if (keys == null) {
      keys = Object.keys(obj);
    }
    for (i = 0, len = keys.length; i < len; i++) {
      k = keys[i];
      if (obj[k] === void 0) {
        console.log('Bad Object: ', obj);
        throw new Error(k + " is undefined");
      }
    }
    return obj;
  };
  KEYS = _.mapObject({
    up: 'uparrow',
    down: 'downarrow',
    right: 'rightarrow',
    left: 'leftarrow'
  }, jsPsych.pluginAPI.convertKeyCharacterToKeyCode);
  KEY_DESCRIPTION = "Navigate with the arrow keys.";
  MouselabMDP = (function() {
    function MouselabMDP(config) {
      this.checkFinished = bind(this.checkFinished, this);
      this.endTrial = bind(this.endTrial, this);
      this.buildMap = bind(this.buildMap, this);
      this.initPlayer = bind(this.initPlayer, this);
      this.startTimer = bind(this.startTimer, this);
      this.draw = bind(this.draw, this);
      this.run = bind(this.run, this);
      this.addScore = bind(this.addScore, this);
      this.arrive = bind(this.arrive, this);
      this.displayFeedback = bind(this.displayFeedback, this);
      this.recordQuery = bind(this.recordQuery, this);
      this.getOutcome = bind(this.getOutcome, this);
      this.getStateLabel = bind(this.getStateLabel, this);
      this.getEdgeLabel = bind(this.getEdgeLabel, this);
      this.mouseoutEdge = bind(this.mouseoutEdge, this);
      this.mouseoverEdge = bind(this.mouseoverEdge, this);
      this.clickEdge = bind(this.clickEdge, this);
      this.mouseoutState = bind(this.mouseoutState, this);
      this.mouseoverState = bind(this.mouseoverState, this);
      this.clickState = bind(this.clickState, this);
      this.handleKey = bind(this.handleKey, this);
      var centerMessage, leftMessage, lowerMessage, ref, ref1, ref10, ref11, ref12, ref13, ref14, ref15, ref16, ref17, ref2, ref3, ref4, ref5, ref6, ref7, ref8, ref9, rightMessage;
      this.display = config.display, this.graph = config.graph, this.layout = config.layout, this.initial = config.initial, this.stateLabels = (ref = config.stateLabels) != null ? ref : null, this.stateDisplay = (ref1 = config.stateDisplay) != null ? ref1 : 'never', this.stateClickCost = (ref2 = config.stateClickCost) != null ? ref2 : PARAMS.info_cost, this.edgeLabels = (ref3 = config.edgeLabels) != null ? ref3 : 'reward', this.edgeDisplay = (ref4 = config.edgeDisplay) != null ? ref4 : 'always', this.edgeClickCost = (ref5 = config.edgeClickCost) != null ? ref5 : 0, this.trial_i = (ref6 = config.trial_i) != null ? ref6 : null, this.stateRewards = (ref7 = config.stateRewards) != null ? ref7 : null, this.keys = (ref8 = config.keys) != null ? ref8 : KEYS, this.trialIndex = (ref9 = config.trialIndex) != null ? ref9 : TRIAL_INDEX, this.playerImage = (ref10 = config.playerImage) != null ? ref10 : 'static/images/plane.png', SIZE = (ref11 = config.SIZE) != null ? ref11 : 120, leftMessage = (ref12 = config.leftMessage) != null ? ref12 : 'Round: 1/1', centerMessage = (ref13 = config.centerMessage) != null ? ref13 : '&nbsp;', rightMessage = (ref14 = config.rightMessage) != null ? ref14 : 'Score: <span id=mouselab-score/>', lowerMessage = (ref15 = config.lowerMessage) != null ? ref15 : KEY_DESCRIPTION, this.minTime = (ref16 = config.minTime) != null ? ref16 : (DEBUG ? 5 : 45), this.feedback = (ref17 = config.feedback) != null ? ref17 : true;
      checkObj(this);
      this.invKeys = _.invert(this.keys);
      this.data = {
        delays: [],
        trial_i: this.trial_i,
        trialIndex: this.trialIndex,
        score: 0,
        path: [],
        rt: [],
        actions: [],
        actionTimes: [],
        queries: {
          click: {
            state: {
              'target': [],
              'time': []
            },
            edge: {
              'target': [],
              'time': []
            }
          },
          mouseover: {
            state: {
              'target': [],
              'time': []
            },
            edge: {
              'target': [],
              'time': []
            }
          },
          mouseout: {
            state: {
              'target': [],
              'time': []
            },
            edge: {
              'target': [],
              'time': []
            }
          }
        }
      };
      this.leftMessage = $('<div>', {
        id: 'mouselab-msg-left',
        "class": 'mouselab-header',
        html: leftMessage
      }).appendTo(this.display);
      this.centerMessage = $('<div>', {
        id: 'mouselab-msg-center',
        "class": 'mouselab-header',
        html: 'Time: <span id=mdp-time/>'
      }).appendTo(this.display);
      this.rightMessage = $('<div>', {
        id: 'mouselab-msg-right',
        "class": 'mouselab-header',
        html: rightMessage
      }).appendTo(this.display);
      this.addScore(0);
      this.canvasElement = $('<canvas>', {
        id: 'mouselab-canvas'
      }).attr({
        width: 500,
        height: 500
      }).appendTo(this.display);
      this.lowerMessage = $('<div>', {
        id: 'mouselab-msg-bottom',
        html: lowerMessage || '&nbsp'
      }).appendTo(this.display);
      mdp = this;
      LOG_INFO('new MouselabMDP', this);
      $('#jspsych-target').append("<div id=\"mdp-feedback\" class=\"modal\">\n  <div id=\"mdp-feedback-content\" class=\"modal-content\">\n    <h3>Default</h3>\n  </div>\n</div>");
    }

    MouselabMDP.prototype.handleKey = function(s0, a) {
      var r, ref, s1, s1g;
      LOG_DEBUG('handleKey', s0, a);
      this.data.actions.push(a);
      this.data.actionTimes.push(Date.now() - this.initTime);
      ref = this.getOutcome(s0, a), r = ref[0], s1 = ref[1];
      LOG_DEBUG(s0 + ", " + a + " -> " + r + ", " + s1);
      s1g = this.states[s1];
      return this.player.animate({
        left: s1g.left,
        top: s1g.top
      }, {
        duration: dist(this.player, s0) * 4,
        onChange: this.canvas.renderAll.bind(this.canvas),
        onComplete: (function(_this) {
          return function() {
            _this.addScore(r);
            return _this.displayFeedback(a, s1);
          };
        })(this)
      });
    };

    MouselabMDP.prototype.clickState = function(g, s) {
      LOG_DEBUG("clickState " + s);
      if (this.stateLabels && this.stateDisplay === 'click' && !g.label.text) {
        this.addScore(-this.stateClickCost);
        g.setLabel(this.getStateLabel(s));
        return this.recordQuery('click', 'state', s);
      }
    };

    MouselabMDP.prototype.mouseoverState = function(g, s) {
      LOG_DEBUG("mouseoverState " + s);
      if (this.stateLabels && this.stateDisplay === 'hover') {
        g.setLabel(this.getStateLabel(s));
        return this.recordQuery('mouseover', 'state', s);
      }
    };

    MouselabMDP.prototype.mouseoutState = function(g, s) {
      LOG_DEBUG("mouseoutState " + s);
      if (this.stateLabels && this.stateDisplay === 'hover') {
        g.setLabel('');
        return this.recordQuery('mouseout', 'state', s);
      }
    };

    MouselabMDP.prototype.clickEdge = function(g, s0, r, s1) {
      LOG_DEBUG("clickEdge " + s0 + " " + r + " " + s1);
      if (this.edgeLabels && this.edgeDisplay === 'click' && !g.label.text) {
        g.setLabel(this.getEdgeLabel(s0, r, s1));
        return this.recordQuery('click', 'edge', s0 + "__" + s1);
      }
    };

    MouselabMDP.prototype.mouseoverEdge = function(g, s0, r, s1) {
      LOG_DEBUG("mouseoverEdge " + s0 + " " + r + " " + s1);
      if (this.edgeLabels && this.edgeDisplay === 'hover') {
        g.setLabel(this.getEdgeLabel(s0, r, s1));
        return this.recordQuery('mouseover', 'edge', s0 + "__" + s1);
      }
    };

    MouselabMDP.prototype.mouseoutEdge = function(g, s0, r, s1) {
      LOG_DEBUG("mouseoutEdge " + s0 + " " + r + " " + s1);
      if (this.edgeLabels && this.edgeDisplay === 'hover') {
        g.setLabel('');
        return this.recordQuery('mouseout', 'edge', s0 + "__" + s1);
      }
    };

    MouselabMDP.prototype.getEdgeLabel = function(s0, r, s1) {
      if (this.edgeLabels === 'reward') {
        return String(r);
      } else {
        return this.edgeLabels[s0 + "__" + s1];
      }
    };

    MouselabMDP.prototype.getStateLabel = function(s) {
      if (this.stateLabels != null) {
        switch (this.stateLabels) {
          case 'custom':
            return ':)';
          case 'reward':
            return this.stateRewards[s];
          default:
            return this.stateLabels[s];
        }
      } else {
        return '';
      }
    };

    MouselabMDP.prototype.getOutcome = function(s0, a) {
      var r, ref, s1;
      ref = this.graph[s0][a], r = ref[0], s1 = ref[1];
      if (this.stateRewards) {
        r = this.stateRewards[s1];
      }
      return [r, s1];
    };

    MouselabMDP.prototype.recordQuery = function(queryType, targetType, target) {
      this.canvas.renderAll();
      LOG_DEBUG("recordQuery " + queryType + " " + targetType + " " + target);
      this.data.queries[queryType][targetType].target.push(target);
      return this.data.queries[queryType][targetType].time.push(Date.now() - this.initTime);
    };

    MouselabMDP.prototype.displayFeedback = function(a, s1) {
      var feedback, head, info, msg, penalty, redGreenSpan, result;
      feedback = registerMove(a);
      console.log('feedback', feedback);
      result = {
        delay: 4
      };
      this.data.delays.push(result.delay);
      redGreenSpan = function(txt, val) {
        return "<span style='color: " + (redGreen(val)) + "; font-weight: bold;'>" + txt + "</span>";
      };
      if (PARAMS.message) {
        head = (function() {
          if (PARAMS.message === 'full') {
            if (result.planned_too_little) {
              if (!result.planned_too_much) {
                return redGreenSpan("You should have gathered more information!", -1);
              } else {
                return redGreenSpan("You gathered too little relevant and too much irrelevant information!", -1);
              }
            } else {
              if (result.planned_too_much) {
                return redGreenSpan("You considered irrelevant outcomes.", -1);
              } else {
                return redGreenSpan("You gathered enough information!", 1);
              }
            }
          } else {
            return redGreenSpan("Poor planning!", -1);
          }
        })();
        penalty = result.delay ? "<p>" + result.delay + " second penalty</p>" : void 0;
        info = (function() {
          if (PARAMS.smart_message) {
            return "Given the information you collected, your decision was " + (result.information_used_correctly ? redGreenSpan('optimal.', 1) : redGreenSpan('suboptimal.', -1));
          } else {
            return '';
          }
        })();
        msg = "<h3>" + head + "</h3>\n<b>" + penalty + "</b>\n" + info;
      } else {
        msg = "Please wait " + result.delay + " seconds.";
      }
      if (this.feedback && result.delay >= 1) {
        this.freeze = true;
        $('#mdp-feedback').css({
          display: 'block'
        });
        $('#mdp-feedback-content').html(msg);
        return setTimeout(((function(_this) {
          return function() {
            _this.freeze = false;
            $('#mdp-feedback').css({
              display: 'none'
            });
            return _this.arrive(s1);
          };
        })(this)), result.delay * 1000);
      } else {
        $('#mdp-feedback').css({
          display: 'none'
        });
        return this.arrive(s1);
      }
    };

    MouselabMDP.prototype.arrive = function(s) {
      var a, keys;
      LOG_DEBUG('arrive', s);
      this.data.path.push(s);
      if (this.graph[s]) {
        keys = (function() {
          var i, len, ref, results;
          ref = Object.keys(this.graph[s]);
          results = [];
          for (i = 0, len = ref.length; i < len; i++) {
            a = ref[i];
            results.push(this.keys[a]);
          }
          return results;
        }).call(this);
      } else {
        keys = [];
      }
      if (!keys.length) {
        this.complete = true;
        this.checkFinished();
        return;
      }
      return this.keyListener = jsPsych.pluginAPI.getKeyboardResponse({
        valid_responses: keys,
        rt_method: 'date',
        persist: false,
        allow_held_key: false,
        callback_function: (function(_this) {
          return function(info) {
            var action;
            action = _this.invKeys[info.key];
            LOG_DEBUG('key', info.key);
            _this.data.rt.push(info.rt);
            return _this.handleKey(s, action);
          };
        })(this)
      });
    };

    MouselabMDP.prototype.addScore = function(v) {
      this.data.score = round(this.data.score + v);
      $('#mouselab-score').html('$' + this.data.score.toFixed(2));
      return $('#mouselab-score').css('color', redGreen(this.data.score));
    };

    MouselabMDP.prototype.run = function() {
      LOG_DEBUG('run');
      meta_MDP.init(this.trial_i);
      this.buildMap();
      this.startTimer();
      return fabric.Image.fromURL(this.playerImage, ((function(_this) {
        return function(img) {
          _this.initPlayer(img);
          _this.canvas.renderAll();
          _this.initTime = Date.now();
          return _this.arrive(_this.initial);
        };
      })(this)));
    };

    MouselabMDP.prototype.draw = function(obj) {
      this.canvas.add(obj);
      return obj;
    };

    MouselabMDP.prototype.startTimer = function() {
      var intervalID, tick;
      this.timeLeft = this.minTime;
      intervalID = void 0;
      tick = (function(_this) {
        return function() {
          if (_this.freeze) {
            return;
          }
          _this.timeLeft -= 1;
          $('#mdp-time').html(_this.timeLeft);
          $('#mdp-time').css('color', redGreen(-_this.timeLeft + .1));
          if (_this.timeLeft === 0) {
            window.clearInterval(intervalID);
            return _this.checkFinished();
          }
        };
      })(this);
      $('#mdp-time').html(this.timeLeft);
      $('#mdp-time').css('color', redGreen(-this.timeLeft + .1));
      return intervalID = window.setInterval(tick, 1000);
    };

    MouselabMDP.prototype.initPlayer = function(img) {
      var left, top;
      LOG_DEBUG('initPlayer');
      top = this.states[this.initial].top;
      left = this.states[this.initial].left;
      img.scale(0.3);
      img.set('top', top).set('left', left);
      this.draw(img);
      return this.player = img;
    };

    MouselabMDP.prototype.buildMap = function() {
      var a, actions, height, location, r, ref, ref1, ref2, results, s, s0, s1, width, x, y;
      ref = (function(_this) {
        return function() {
          var ref, xs, ys;
          ref = _.unzip(_.values(_this.layout)), xs = ref[0], ys = ref[1];
          return [(_.max(xs)) + 1, (_.max(ys)) + 1];
        };
      })(this)(), width = ref[0], height = ref[1];
      this.canvasElement.attr({
        width: width * SIZE,
        height: height * SIZE
      });
      this.canvas = new fabric.Canvas('mouselab-canvas', {
        selection: false
      });
      this.states = {};
      ref1 = this.layout;
      for (s in ref1) {
        location = ref1[s];
        x = location[0], y = location[1];
        this.states[s] = this.draw(new State(s, x, y, {
          fill: '#bbb',
          label: this.stateDisplay === 'always' ? this.getStateLabel(s) : ''
        }));
      }
      LOG_DEBUG('@graph', this.graph);
      LOG_DEBUG('@states', this.states);
      ref2 = this.graph;
      results = [];
      for (s0 in ref2) {
        actions = ref2[s0];
        results.push((function() {
          var ref3, results1;
          results1 = [];
          for (a in actions) {
            ref3 = actions[a], r = ref3[0], s1 = ref3[1];
            results1.push(this.draw(new Edge(this.states[s0], r, this.states[s1], {
              label: this.edgeDisplay === 'always' ? this.getEdgeLabel(s0, r, s1) : ''
            })));
          }
          return results1;
        }).call(this));
      }
      return results;
    };

    MouselabMDP.prototype.endTrial = function() {
      this.lowerMessage.html("<b>Press any key to continue.</br>");
      return this.keyListener = jsPsych.pluginAPI.getKeyboardResponse({
        valid_responses: [],
        rt_method: 'date',
        persist: false,
        allow_held_key: false,
        callback_function: (function(_this) {
          return function(info) {
            _this.display.empty();
            return jsPsych.finishTrial(_this.data);
          };
        })(this)
      });
    };

    MouselabMDP.prototype.checkFinished = function() {
      if (this.complete && (this.timeLeft != null) && this.timeLeft > 0) {
        this.lowerMessage.html("Waiting for the timer to expire...");
      }
      if (this.complete && this.timeLeft <= 0) {
        return this.endTrial();
      }
    };

    return MouselabMDP;

  })();
  State = (function(superClass) {
    extend(State, superClass);

    function State(name, left, top, config) {
      var conf;
      this.name = name;
      if (config == null) {
        config = {};
      }
      left = (left + 0.5) * SIZE;
      top = (top + 0.5) * SIZE;
      conf = {
        left: left,
        top: top,
        fill: '#bbbbbb',
        radius: SIZE / 4,
        label: ''
      };
      _.extend(conf, config);
      this.circle = new fabric.Circle(conf);
      this.label = new Text('----------', left, top, {
        fontSize: SIZE / 6,
        fill: '#44d'
      });
      this.radius = this.circle.radius;
      this.left = this.circle.left;
      this.top = this.circle.top;
      this.on('mousedown', function() {
        return mdp.clickState(this, this.name);
      });
      this.on('mouseover', function() {
        return mdp.mouseoverState(this, this.name);
      });
      this.on('mouseout', function() {
        return mdp.mouseoutState(this, this.name);
      });
      State.__super__.constructor.call(this, [this.circle, this.label]);
      this.setLabel(conf.label);
    }

    State.prototype.setLabel = function(txt) {
      if (txt) {
        this.label.setText("" + txt);
        this.label.setFill(redGreen(txt));
      } else {
        this.label.setText('');
      }
      return this.dirty = true;
    };

    return State;

  })(fabric.Group);
  Edge = (function(superClass) {
    extend(Edge, superClass);

    function Edge(c1, reward, c2, config) {
      var adjX, adjY, ang, labX, labY, label, ref, ref1, ref2, ref3, ref4, ref5, ref6, rotateLabel, spacing, x1, x2, y1, y2;
      if (config == null) {
        config = {};
      }
      spacing = (ref = config.spacing) != null ? ref : 8, adjX = (ref1 = config.adjX) != null ? ref1 : 0, adjY = (ref2 = config.adjY) != null ? ref2 : 0, rotateLabel = (ref3 = config.rotateLabel) != null ? ref3 : false, label = (ref4 = config.label) != null ? ref4 : '';
      ref5 = [c1.left + adjX, c1.top + adjY, c2.left + adjX, c2.top + adjY], x1 = ref5[0], y1 = ref5[1], x2 = ref5[2], y2 = ref5[3];
      this.arrow = new Arrow(x1, y1, x2, y2, c1.radius + spacing, c2.radius + spacing);
      ang = (this.arrow.ang + Math.PI / 2) % (Math.PI * 2);
      if ((0.5 * Math.PI <= ang && ang <= 1.5 * Math.PI)) {
        ang += Math.PI;
      }
      ref6 = polarMove(x1, y1, angle(x1, y1, x2, y2), SIZE * 0.45), labX = ref6[0], labY = ref6[1];
      this.label = new Text('----------', labX, labY, {
        angle: rotateLabel ? ang * 180 / Math.PI : 0,
        fill: redGreen(label),
        fontSize: SIZE / 6,
        textBackgroundColor: 'white'
      });
      this.on('mousedown', function() {
        return mdp.clickEdge(this, c1.name, reward, c2.name);
      });
      this.on('mouseover', function() {
        return mdp.mouseoverEdge(this, c1.name, reward, c2.name);
      });
      this.on('mouseout', function() {
        return mdp.mouseoutEdge(this, c1.name, reward, c2.name);
      });
      Edge.__super__.constructor.call(this, [this.arrow, this.label]);
      this.setLabel(label);
    }

    Edge.prototype.setLabel = function(txt) {
      if (txt) {
        this.label.setText("" + txt);
        this.label.setFill(redGreen(txt));
      } else {
        this.label.setText('');
      }
      return this.dirty = true;
    };

    return Edge;

  })(fabric.Group);
  Arrow = (function(superClass) {
    extend(Arrow, superClass);

    function Arrow(x1, y1, x2, y2, adj1, adj2) {
      var ang, deltaX, deltaY, dx, dy, line, point, ref, ref1;
      if (adj1 == null) {
        adj1 = 0;
      }
      if (adj2 == null) {
        adj2 = 0;
      }
      this.ang = ang = angle(x1, y1, x2, y2);
      ref = polarMove(x1, y1, ang, adj1), x1 = ref[0], y1 = ref[1];
      ref1 = polarMove(x2, y2, ang, -(adj2 + 7.5)), x2 = ref1[0], y2 = ref1[1];
      line = new fabric.Line([x1, y1, x2, y2], {
        stroke: '#555',
        selectable: false,
        strokeWidth: 3
      });
      this.centerX = (x1 + x2) / 2;
      this.centerY = (y1 + y2) / 2;
      deltaX = line.left - this.centerX;
      deltaY = line.top - this.centerY;
      dx = x2 - x1;
      dy = y2 - y1;
      point = new fabric.Triangle({
        left: x2 + deltaX,
        top: y2 + deltaY,
        pointType: 'arrow_start',
        angle: ang * 180 / Math.PI,
        width: 10,
        height: 10,
        fill: '#555'
      });
      Arrow.__super__.constructor.call(this, [line, point]);
    }

    return Arrow;

  })(fabric.Group);
  Text = (function(superClass) {
    extend(Text, superClass);

    function Text(txt, left, top, config) {
      var conf;
      txt = String(txt);
      conf = {
        left: left,
        top: top,
        fontFamily: 'helvetica',
        fontSize: SIZE / 8
      };
      _.extend(conf, config);
      Text.__super__.constructor.call(this, txt, conf);
    }

    return Text;

  })(fabric.Text);
  plugin = {
    trial: function(display_element, trialConfig) {
      var trial;
      trialConfig = jsPsych.pluginAPI.evaluateFunctionParameters(trialConfig);
      trialConfig.display = display_element;
      console.log('trialConfig', trialConfig);
      display_element.empty();
      trial = new MouselabMDP(trialConfig);
      trial.run();
      if (trialConfig._block) {
        trialConfig._block.trialCount += 1;
      }
      return TRIAL_INDEX += 1;
    }
  };
  return plugin;
})();
