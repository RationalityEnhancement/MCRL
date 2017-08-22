// Generated by CoffeeScript 1.12.3

/*
jspsych-mouselab-mdp.coffee
Fred Callaway

https://github.com/fredcallaway/Mouselab-MDP
 */
var OPTIMAL, TRIAL_INDEX, mdp,
  slice = [].slice,
  bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; },
  extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
  hasProp = {}.hasOwnProperty;

mdp = void 0;

OPTIMAL = void 0;

TRIAL_INDEX = 1;

jsPsych.plugins['mouselab-mdp'] = (function() {
  var Arrow, DEMO_SPEED, Edge, KEYS, LOG_DEBUG, LOG_INFO, MOVE_SPEED, MouselabMDP, NULL, PRINT, SIZE, State, TERM_ACTION, Text, UNKNOWN, angle, checkObj, dist, plugin, polarMove, redGreen, round;
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
  DEMO_SPEED = 1000;
  MOVE_SPEED = 500;
  UNKNOWN = '__';
  TERM_ACTION = '__TERM_ACTION__';
  fabric.Object.prototype.originX = fabric.Object.prototype.originY = 'center';
  fabric.Object.prototype.selectable = false;
  fabric.Object.prototype.hoverCursor = 'plain';
  if (SHOW_PARTICIPANT_DATA) {
    OPTIMAL = loadJson("static/json/data/1B.0/traces/" + SHOW_PARTICIPANT_DATA + ".json");
    DEMO_SPEED = 500;
    MOVE_SPEED = 300;
  } else {
    OPTIMAL = (loadJson('static/json/optimal_policy.json'))[COST_LEVEL];
  }
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
    var j, k, len;
    if (keys == null) {
      keys = Object.keys(obj);
    }
    for (j = 0, len = keys.length; j < len; j++) {
      k = keys[j];
      if (obj[k] === void 0) {
        console.log('Bad Object: ', obj);
        throw new Error(k + " is undefined");
      }
    }
    return obj;
  };
  KEYS = _.mapObject({
    2: 'uparrow',
    0: 'downarrow',
    1: 'rightarrow',
    3: 'leftarrow'
  }, jsPsych.pluginAPI.convertKeyCharacterToKeyCode);
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
      this.getStateLabel = bind(this.getStateLabel, this);
      this.getEdgeLabel = bind(this.getEdgeLabel, this);
      this.mouseoutEdge = bind(this.mouseoutEdge, this);
      this.mouseoverEdge = bind(this.mouseoverEdge, this);
      this.clickEdge = bind(this.clickEdge, this);
      this.mouseoutState = bind(this.mouseoutState, this);
      this.mouseoverState = bind(this.mouseoverState, this);
      this.clickState = bind(this.clickState, this);
      this.handleKey = bind(this.handleKey, this);
      this.runDemo = bind(this.runDemo, this);
      var centerMessage, leftMessage, lowerMessage, ref, ref1, ref10, ref11, ref12, ref13, ref14, ref15, ref16, ref17, ref18, ref19, ref2, ref3, ref4, ref5, ref6, ref7, ref8, ref9, rightMessage, timeMsg;
      this.display = config.display, this.graph = config.graph, this.layout = config.layout, this.tree = (ref = config.tree) != null ? ref : null, this.initial = config.initial, this.stateLabels = (ref1 = config.stateLabels) != null ? ref1 : null, this.stateDisplay = (ref2 = config.stateDisplay) != null ? ref2 : 'never', this.stateClickCost = (ref3 = config.stateClickCost) != null ? ref3 : PARAMS.info_cost, this.edgeLabels = (ref4 = config.edgeLabels) != null ? ref4 : 'reward', this.edgeDisplay = (ref5 = config.edgeDisplay) != null ? ref5 : 'always', this.edgeClickCost = (ref6 = config.edgeClickCost) != null ? ref6 : 0, this.trial_i = (ref7 = config.trial_i) != null ? ref7 : null, this.demonstrate = (ref8 = config.demonstrate) != null ? ref8 : false, this.stateRewards = (ref9 = config.stateRewards) != null ? ref9 : null, this.keys = (ref10 = config.keys) != null ? ref10 : KEYS, this.trialIndex = (ref11 = config.trialIndex) != null ? ref11 : TRIAL_INDEX, this.playerImage = (ref12 = config.playerImage) != null ? ref12 : 'static/images/plane.png', SIZE = (ref13 = config.SIZE) != null ? ref13 : 110, leftMessage = (ref14 = config.leftMessage) != null ? ref14 : null, centerMessage = (ref15 = config.centerMessage) != null ? ref15 : '&nbsp;', rightMessage = (ref16 = config.rightMessage) != null ? ref16 : 'Score: <span id=mouselab-score/>', lowerMessage = (ref17 = config.lowerMessage) != null ? ref17 : "Navigate with the arrow keys.", this.minTime = (ref18 = config.minTime) != null ? ref18 : (DEBUG ? 5 : 45), this.feedback = (ref19 = config.feedback) != null ? ref19 : true;
      this.initial = 0;
      this.tree = [[1, 5, 9, 13], [2], [3, 4], [], [], [6], [7, 8], [], [], [10], [11, 12], [], [], [14], [15, 16], [], []];
      this.transition = [
        {
          '0': 1,
          '1': 5,
          '2': 9,
          '3': 13
        }, {
          '0': 2
        }, {
          '1': 3,
          '3': 4
        }, {}, {}, {
          '1': 6
        }, {
          '0': 8,
          '2': 7
        }, {}, {}, {
          '2': 10
        }, {
          '1': 12,
          '3': 11
        }, {}, {}, {
          '3': 14
        }, {
          '0': 15,
          '2': 16
        }, {}, {}
      ];
      this.layout = [[0, 0], [0, 1], [0, 2], [1, 2], [-1, 2], [1, 0], [2, 0], [2, -1], [2, 1], [0, -1], [0, -2], [-1, -2], [1, -2], [-1, 0], [-2, 0], [-2, 1], [-2, -1]];
      if (leftMessage == null) {
        leftMessage = "Round: " + TRIAL_INDEX + "/" + N_TRIALS;
      }
      if (this.demonstrate) {
        lowerMessage = "This is a demonstration of optimal planning.";
      }
      if (SHOW_PARTICIPANT_DATA) {
        this.demonstrate = true;
        lowerMessage = "Behavior of participant " + SHOW_PARTICIPANT_DATA;
      }
      console.log('TRIAL NUMBER', this.trial_i);
      checkObj(this);
      this.initial = "" + this.initial;
      this.invKeys = _.invert(this.keys);
      this.data = {
        delays: [],
        planned_too_little: [],
        planned_too_much: [],
        information_used_correctly: [],
        trial_i: this.trial_i,
        trialIndex: this.trialIndex,
        score: 0,
        path: [],
        rt: [],
        actions: [],
        actionTimes: [],
        beliefs: [],
        metaActions: [],
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
      timeMsg = this.demonstrate ? '&nbsp;' : 'Time: <span id=mdp-time/>';
      this.centerMessage = $('<div>', {
        id: 'mouselab-msg-center',
        "class": 'mouselab-header',
        html: timeMsg
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

    MouselabMDP.prototype.runDemo = function() {
      var actions, i, interval;
      this.feedback = false;
      this.timeLeft = 1;
      actions = OPTIMAL[this.trial_i];
      i = 0;
      return interval = ifvisible.onEvery(1, (function(_this) {
        return function() {
          var a, s;
          if (ifvisible.now()) {
            a = actions[i];
            if (a.is_click) {
              _this.clickState(_this.states[a.state], a.state);
            } else {
              s = _.last(_this.data.path);
              _this.handleKey(s, a.move);
            }
            i += 1;
            if (i === actions.length) {
              return interval.stop();
            }
          }
        };
      })(this));
    };

    MouselabMDP.prototype.handleKey = function(s0, a) {
      var r, s1, s1g;
      LOG_DEBUG('handleKey', s0, a);
      this.data.actions.push(a);
      this.data.actionTimes.push(Date.now() - this.initTime);
      if (!this.disableClicks) {
        this.updatePR(TERM_ACTION);
        this.disableClicks = true;
      }
      s1 = this.transition[s0][a];
      r = this.stateRewards[s1];
      LOG_DEBUG(s0 + ", " + a + " -> " + r + ", " + s1);
      s1g = this.states[s1];
      return this.player.animate({
        left: s1g.left,
        top: s1g.top
      }, {
        duration: MOVE_SPEED,
        onChange: this.canvas.renderAll.bind(this.canvas),
        onComplete: (function(_this) {
          return function() {
            _this.addScore(r);
            if (_this.feedback) {
              return _this.displayFeedback(a, s1);
            } else {
              return _this.arrive(s1);
            }
          };
        })(this)
      });
    };

    MouselabMDP.prototype.clickState = function(g, s) {
      var r;
      if (this.disableClicks) {
        return;
      }
      LOG_DEBUG("clickState " + s);
      if (this.complete || s === this.initial) {
        return;
      }
      if (this.stateLabels && this.stateDisplay === 'click' && !g.label.text) {
        this.addScore(-this.stateClickCost);
        r = this.getStateLabel(s);
        g.setLabel(r);
        this.recordQuery('click', 'state', s);
        return delay(0, (function(_this) {
          return function() {
            return _this.updatePR(parseInt(s), r);
          };
        })(this));
      }
    };

    MouselabMDP.prototype.updatePR = function(action, r) {
      var state;
      state = this.beliefState.slice();
      this.data.beliefs.push(state);
      this.PR = this.PR.then((function(_this) {
        return function(prevPR) {
          var arg;
          arg = {
            state: state,
            action: action
          };
          return callWebppl('PR', arg).then(function(newPR) {
            return prevPR + newPR;
          });
        };
      })(this));
      if (action !== TERM_ACTION) {
        return this.beliefState[action] = r;
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

    MouselabMDP.prototype.recordQuery = function(queryType, targetType, target) {
      this.canvas.renderAll();
      LOG_DEBUG("recordQuery " + queryType + " " + targetType + " " + target);
      this.data.queries[queryType][targetType].target.push(target);
      return this.data.queries[queryType][targetType].time.push(Date.now() - this.initTime);
    };

    MouselabMDP.prototype.displayFeedback = function(a, s1) {
      var head, info, msg, penalty, redGreenSpan, showCriticism;
      this.PR.then((function(_this) {
        return function(pr) {
          console.log("total PR = " + pr);
          return _this.arrive(s1);
        };
      })(this));
      return;
      if (!this.feedback) {
        $('#mdp-feedback').css({
          display: 'none'
        });
        this.arrive(s1);
        return;
      }
      result.delay = Math.round(result.delay);
      console.log('feedback', result);
      showCriticism = result.delay >= 1;
      if (PARAMS.PR_type === 'none') {
        result.delay = (function() {
          switch (PARAMS.info_cost) {
            case 0.01:
              return [null, 4, 0, 1][this.data.actions.length];
            case 1.00:
              return [null, 3, 0, 1][this.data.actions.length];
            case 2.50:
              return [null, 15, 0, 3][this.data.actions.length];
            case 1.0001:
              return [null, 2, 0, 1][this.data.actions.length];
          }
        }).call(this);
      }
      this.data.delays.push(result.delay);
      this.data.planned_too_little.push(result.planned_too_little);
      this.data.planned_too_much.push(result.planned_too_much);
      this.data.information_used_correctly.push(result.information_used_correctly);
      redGreenSpan = function(txt, val) {
        return "<span style='color: " + (redGreen(val)) + "; font-weight: bold;'>" + txt + "</span>";
      };
      if (PARAMS.message) {
        if (PARAMS.PR_type === 'objectLevel') {
          if (a === result.optimal_action.direction) {
            head = redGreenSpan("You chose the best possible move.", 1);
          } else {
            head = redGreenSpan("Bad move! You should have moved " + result.optimal_action.direction + ".", -1);
          }
        } else {
          if (PARAMS.message === 'full') {
            if (result.planned_too_little && showCriticism) {
              if (result.planned_too_much && showCriticism) {
                head = redGreenSpan("You gathered the wrong information.", -1);
              } else {
                head = redGreenSpan("You gathered too little information.", -1);
              }
            } else {
              if (result.planned_too_much && showCriticism) {
                head = redGreenSpan("You gathered too much information.", -1);
              } else {
                if (!result.planned_too_much & !result.planned_too_little) {
                  head = redGreenSpan("You gathered the right amount of information.", 1);
                }
                if (result.information_used_correctly && showCriticism) {
                  head += redGreenSpan(" But you didn't prioritize the most important locations.", -1);
                }
              }
            }
          }
          if (PARAMS.message === 'simple') {
            head = '';
          }
          if (PARAMS.message === 'none') {
            if (result.delay === 1) {
              head = "Please wait 1 second.";
            } else {
              head = "Please wait " + result.delay + " seconds.";
            }
          }
        }
      }
      if (PARAMS.PR_type === "none") {
        penalty = result.delay ? "<p>Please wait " + result.delay + " seconds.</p>" : void 0;
      } else {
        penalty = result.delay ? redGreenSpan("<p>" + result.delay + " second penalty!</p>", -1) : void 0;
      }
      info = (function() {
        if (PARAMS.message === 'full') {
          return "Given the information you collected, your decision was " + (result.information_used_correctly ? redGreenSpan('optimal.', 1) : redGreenSpan('suboptimal.', -1));
        } else {
          return '';
        }
      })();
      if ((PARAMS.message === 'full' || PARAMS.message === 'simple') && PARAMS.PR_type !== 'objectLevel') {
        msg = "<h3>" + head + "</h3>            \n<b>" + penalty + "</b>                        \n" + info;
      }
      if (PARAMS.PR_type === 'objectLevel') {
        msg = "<h3>" + head + "</h3>             \n<b>" + penalty + "</b> ";
      }
      if (PARAMS.message === 'none') {
        msg = "<h3>" + head + "</h3>";
      }
      if (!PARAMS.message) {
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
        })(this)), (false ? 1000 : result.delay * 1000));
      } else {
        $('#mdp-feedback').css({
          display: 'none'
        });
        return this.arrive(s1);
      }
    };

    MouselabMDP.prototype.arrive = function(s) {
      var a, keys;
      this.PR = new Promise(function(resolve) {
        return resolve(0);
      });
      LOG_DEBUG('arrive', s);
      this.data.path.push(s);
      if (this.transition[s]) {
        keys = (function() {
          var j, len, ref, results;
          ref = Object.keys(this.transition[s]);
          results = [];
          for (j = 0, len = ref.length; j < len; j++) {
            a = ref[j];
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
      if (!this.demonstrate) {
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
      }
    };

    MouselabMDP.prototype.addScore = function(v) {
      this.data.score = round(this.data.score + v);
      $('#mouselab-score').html('$' + this.data.score.toFixed(2));
      return $('#mouselab-score').css('color', redGreen(this.data.score));
    };

    MouselabMDP.prototype.run = function() {
      LOG_DEBUG('run');
      this.buildMap();
      this.startTimer();
      return fabric.Image.fromURL(this.playerImage, ((function(_this) {
        return function(img) {
          _this.initPlayer(img);
          _this.canvas.renderAll();
          _this.initTime = Date.now();
          _this.arrive(_this.initial);
          if (_this.demonstrate) {
            return _this.runDemo();
          }
        };
      })(this)));
    };

    MouselabMDP.prototype.draw = function(obj) {
      this.canvas.add(obj);
      return obj;
    };

    MouselabMDP.prototype.startTimer = function() {
      var interval, intervalID;
      this.timeLeft = this.minTime;
      intervalID = void 0;
      interval = ifvisible.onEvery(1, (function(_this) {
        return function() {
          if (_this.freeze) {
            return;
          }
          _this.timeLeft -= 1;
          $('#mdp-time').html(_this.timeLeft);
          $('#mdp-time').css('color', redGreen(-_this.timeLeft + .1));
          if (_this.timeLeft === 0) {
            interval.stop();
            return _this.checkFinished();
          }
        };
      })(this));
      $('#mdp-time').html(this.timeLeft);
      return $('#mdp-time').css('color', redGreen(-this.timeLeft + .1));
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
      (function(_this) {
        return (function() {
          var height, minx, miny, ref, width, xs, ys;
          ref = _.unzip(_.values(_this.layout)), xs = ref[0], ys = ref[1];
          minx = _.min(xs);
          miny = _.min(ys);
          xs = xs.map(function(x) {
            return x - minx;
          });
          ys = ys.map(function(y) {
            return y - miny;
          });
          _this.layout = _.zip(xs, ys);
          width = (_.max(xs)) + 1;
          height = (_.max(ys)) + 1;
          return _this.canvasElement.attr({
            width: width * SIZE,
            height: height * SIZE
          });
        });
      })(this)();
      this.canvas = new fabric.Canvas('mouselab-canvas', {
        selection: false
      });
      this.states = [];
      this.beliefState = [];
      this.layout.forEach((function(_this) {
        return function(loc, idx) {
          var x, y;
          _this.beliefState.push(UNKNOWN);
          x = loc[0], y = loc[1];
          return _this.states.push(_this.draw(new State(idx, x, y, {
            fill: '#bbb',
            label: ''
          })));
        };
      })(this));
      this.beliefState[0] = 0;
      this.data.beliefs.push(this.beliefState);
      LOG_INFO('@states', this.states);
      return this.tree.forEach((function(_this) {
        return function(s1s, s0) {
          return s1s.forEach(function(s1) {
            return _this.draw(new Edge(_this.states[s0], 0, _this.states[s1], {
              label: ''
            }));
          });
        };
      })(this));
    };

    MouselabMDP.prototype.endTrial = function() {
      SCORE += this.data.score;
      this.lowerMessage.html("So far, you've earned a bonus of $" + (calculateBonus().toFixed(2)) + "\n<br>\n<b>Press any key to continue.</b><e");
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
      if (!mdp.demonstrate) {
        this.on('mousedown', function() {
          return mdp.clickState(this, this.name);
        });
        this.on('mouseover', function() {
          return mdp.mouseoverState(this, this.name);
        });
        this.on('mouseout', function() {
          return mdp.mouseoutState(this, this.name);
        });
      }
      State.__super__.constructor.call(this, [this.circle, this.label]);
      this.setLabel(conf.label);
    }

    State.prototype.setLabel = function(txt) {
      if ("" + txt) {
        this.label.setText("$" + txt);
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
