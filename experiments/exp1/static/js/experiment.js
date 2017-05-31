// Generated by CoffeeScript 1.12.3

/*
experiment.coffee
Fred Callaway

Demonstrates the jsych-mdp plugin
 */
var BLOCKS, DEBUG, DEMO, IVs, N_TRIALS, PARAMS, PRType, TRIALS, condition, conditions, counterbalance, createStartButton, delay, experiment_nr, i, infoCost, initializeExperiment, j, k, len, len1, len2, message, messageTypes, nrConditions, nrDelays, nrInfoCosts, nrMessages, psiturk, ref, ref1,
  extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
  hasProp = {}.hasOwnProperty;

DEBUG = true;

experiment_nr = 0;

switch (experiment_nr) {
  case 0:
    IVs = {
      PRTypes: ['none', 'featureBased', 'fullObservation'],
      messageTypes: ['full', 'none'],
      infoCosts: [1.60]
    };
    break;
  case 1:
    IVs = {
      PRTypes: ['none', 'featureBased', 'fullObservation'],
      messageTypes: ['full', 'none'],
      infoCosts: [0.01, 1.60, 2.80]
    };
    break;
  case 2:
    IVs = {
      PRTypes: ['featureBased', 'objectLevel'],
      messageTypes: ['full'],
      infoCosts: [0.01, 1.60, 2.80]
    };
    break;
  case 3:
    IVs = {
      PRTypes: ['none', 'featureBased'],
      messageTypes: ['full', 'simple'],
      infoCosts: [1.60]
    };
    break;
  default:
    console.log("Invalid experiment_nr!");
}

nrDelays = IVs.PRTypes.length;

nrMessages = IVs.messageTypes.length;

nrInfoCosts = IVs.infoCosts.length;

nrConditions = (function() {
  switch (experiment_nr) {
    case 0:
      return 3;
    case 1:
      return 3 * 3;
    default:
      return nrDelays * nrMessages * nrInfoCosts;
  }
})();

conditions = {
  'PRType': [],
  'messageType': [],
  'infoCost': []
};

ref = IVs.PRTypes;
for (i = 0, len = ref.length; i < len; i++) {
  PRType = ref[i];
  if (experiment_nr <= 1) {
    if (PRType === 'none') {
      messageTypes = ['none'];
    } else {
      messageTypes = ['full'];
    }
  } else {
    messageTypes = IVs.messageTypes;
  }
  for (j = 0, len1 = messageTypes.length; j < len1; j++) {
    message = messageTypes[j];
    ref1 = IVs.infoCosts;
    for (k = 0, len2 = ref1.length; k < len2; k++) {
      infoCost = ref1[k];
      conditions.PRType.push(PRType);
      conditions.messageType.push(message);
      conditions.infoCost.push(infoCost);
    }
  }
}

if (DEBUG) {
  console.log("X X X X X X X X X X X X X X X X X\n X X X X X DEBUG  MODE X X X X X\nX X X X X X X X X X X X X X X X X");
  condition = 2;
} else {
  console.log("# =============================== #\n# ========= NORMAL MODE ========= #\n# =============================== #");
}

if (mode === "{{ mode }}") {
  DEMO = true;
  condition = 1;
  counterbalance = 0;
}

psiturk = new PsiTurk(uniqueId, adServerLoc, mode);

BLOCKS = void 0;

PARAMS = void 0;

TRIALS = void 0;

N_TRIALS = void 0;

delay = function(time, func) {
  return setTimeout(func, time);
};

$(window).on('load', function() {
  var loadTimeout, slowLoad;
  slowLoad = function() {
    return document.getElementById("failLoad").style.display = "block";
  };
  loadTimeout = delay(12000, slowLoad);
  psiturk.preloadImages(['static/images/example1.png', 'static/images/example2.png', 'static/images/example3.png', 'static/images/money.png', 'static/images/plane.png', 'static/images/spider.png']);
  return delay(300, function() {
    var ERROR, condition_nr, expData;
    console.log('Loading data');
    expData = loadJson("static/json/condition_0_0.json");
    console.log('expData', expData);
    condition_nr = condition % nrConditions;
    PARAMS = {
      'PR_type': conditions.PRType[condition_nr],
      'feedback': conditions.PRType[condition_nr] !== "none",
      'info_cost': conditions.infoCost[condition_nr],
      'message': conditions.messageType[condition_nr]
    };
    BLOCKS = expData.blocks;
    TRIALS = BLOCKS.standard;
    psiturk.recordUnstructuredData('params', PARAMS);
    psiturk.recordUnstructuredData('experiment_nr', experiment_nr);
    psiturk.recordUnstructuredData('condition_nr', condition_nr);
    if (DEBUG || DEMO) {
      return createStartButton();
    } else {
      console.log('Testing saveData');
      ERROR = null;
      return psiturk.saveData({
        error: function() {
          console.log('ERROR saving data.');
          return ERROR = true;
        },
        success: function() {
          console.log('Data saved to psiturk server.');
          clearTimeout(loadTimeout);
          return delay(500, createStartButton);
        }
      });
    }
  });
});

createStartButton = function() {
  if (DEBUG) {
    initializeExperiment();
    return;
  }
  document.getElementById("loader").style.display = "none";
  document.getElementById("successLoad").style.display = "block";
  document.getElementById("failLoad").style.display = "none";
  return $('#load-btn').click(initializeExperiment);
};

initializeExperiment = function() {
  var BONUS, Block, MDPBlock, QuizLoop, TextBlock, calculateBonus, costLevel, debug_slide, experiment_timeline, finish, instruct_loop, instructions, main, prompt_resubmit, quiz, reprompt, save_data, text;
  console.log('INITIALIZE EXPERIMENT');
  N_TRIALS = BLOCKS.standard.length;
  costLevel = (function() {
    switch (PARAMS.info_cost) {
      case 0.01:
        return 'low';
      case 1.60:
        return 'med';
      case 2.80:
        return 'high';
      default:
        throw new Error('bad info_cost');
    }
  })();
  text = {
    debug: function() {
      if (DEBUG) {
        return "`DEBUG`";
      } else {
        return '';
      }
    },
    feedback: function() {
      if (PARAMS.PR_type !== "none") {
        return [markdown("# Instructions\n\n<b>You will receive feedback about your planning. This feedback will\nhelp you learn how to make better decisions.</b> After each flight, if\nyou did not plan optimally, a feedback message will apear. This message\nwill tell you two things:\n\n1. Whether you observed too few relevant values or if you observed\n   irrelevant values (values of locations that you cant fly to).\n2. Whether you flew along the best route given your current location and\n   the information you had about the values of other locations.\n\nIn the example below, not enough relevant values were observed, and\nas a result there is a 15 second timeout penalty. <b>The duration of\nthe timeout penalty is proportional to how poorly you planned your\nroute:</b> the more money you could have earned from observing more\nvalues and/or choosing a better route, the longer the delay. <b>If\nyou perform optimally, no feedback will be shown and you can proceed\nimmediately.</b> The example message here is not necessarily representative of the feedback you'll receive.\n\n" + (img('task_images/Slide4.png')) + "\n")];
      } else {
        return [];
      }
    },
    constantDelay: function() {
      if (PARAMS.PR_type !== "none") {
        return "";
      } else {
        return "Note: there will be short delays after taking some flights.";
      }
    }
  };
  Block = (function() {
    function Block(config) {
      _.extend(this, config);
      this._block = this;
      if (this._init != null) {
        this._init();
      }
    }

    return Block;

  })();
  TextBlock = (function(superClass) {
    extend(TextBlock, superClass);

    function TextBlock() {
      return TextBlock.__super__.constructor.apply(this, arguments);
    }

    TextBlock.prototype.type = 'text';

    TextBlock.prototype.cont_key = ['space'];

    return TextBlock;

  })(Block);
  QuizLoop = (function(superClass) {
    extend(QuizLoop, superClass);

    function QuizLoop() {
      return QuizLoop.__super__.constructor.apply(this, arguments);
    }

    QuizLoop.prototype.loop_function = function(data) {
      var c, l, len3, ref2;
      console.log('data', data);
      ref2 = data[data.length].correct;
      for (l = 0, len3 = ref2.length; l < len3; l++) {
        c = ref2[l];
        if (!c) {
          return true;
        }
      }
      return false;
    };

    return QuizLoop;

  })(Block);
  MDPBlock = (function(superClass) {
    extend(MDPBlock, superClass);

    function MDPBlock() {
      return MDPBlock.__super__.constructor.apply(this, arguments);
    }

    MDPBlock.prototype.type = 'mouselab-mdp';

    MDPBlock.prototype._init = function() {
      return this.trialCount = 0;
    };

    return MDPBlock;

  })(Block);
  debug_slide = new Block({
    type: 'html',
    url: 'test.html'
  });
  instructions = new Block({
    type: "instructions",
    pages: [markdown("# Instructions " + (text.debug()) + "\n\nIn this game, you are in charge of flying an aircraft. As shown below,\nyou will begin in the central location. The arrows show which actions\nare available in each location. Note that once you have made a move you\ncannot go back; you can only move forward along the arrows. There are\neight possible final destinations labelled 1-8 in the image below. On\nyour way there, you will visit two intermediate locations. <b>Every\nlocation you visit will add or subtract money to your account</b>, and\nyour task is to earn as much money as possible. <b>To find out how much\nmoney you earn or lose in a location, you have to click on it.</b> You\ncan uncover the value of as many or as few locations as you wish.\n\n" + (img('task_images/Slide1.png')) + "\n\nTo navigate the airplane, use the arrows (the example above is non-interactive).\nYou can uncover the value of a location at any time. Click \"Next\" to proceed."), markdown("# Instructions\n\nYou will play the game for " + N_TRIALS + " rounds. The value of every location will\nchange from each round to the next. At the begining of each round, the\nvalue of every location will be hidden, and you will only discover the\nvalue of the locations you click on. The example below shows the value\nof every location, just to give you an example of values you could see\nif you clicked on every location. <b>Every time you click a circle to\nobserve its value, you pay a fee of " + (fmtMoney(PARAMS.info_cost)) + ".</b>\n\n" + (img('task_images/Slide2_' + costLevel + '.png')) + "\n\nEach time you move to a\nlocation, your profit will be adjusted. If you move to a location with\na hidden value, your profit will still be adjusted according to the\nvalue of that location. " + (text.constantDelay()))].concat((text.feedback()).concat([markdown("# Instructions\n\nThere are two more important things to understand:\n1. You must spend at least 45 seconds on each round. A countdown timer\n   will show you how much more time you must spend on the round. You\n   won’t be able to proceed to the next round before the countdown has\n   finished, but you can take as much time as you like afterwards.\n2. </b>You will earn <u>real money</u> for your flights.</b> Specifically,\n   one of the " + N_TRIALS + " rounds will be chosen at random and you will receive 5%\n   of your earnings in that round as a bonus payment.\n\n" + (img('task_images/Slide3.png')) + "\n\n You may proceed to take an entry quiz, or go back to review the instructions.")])),
    show_clickable_nav: true
  });
  quiz = new Block({
    preamble: function() {
      return markdown("# Quiz");
    },
    type: 'survey-multi-choice',
    questions: ["True or false: The hidden values will change each time I start a new round.", "How much does it cost to observe each hidden value?", "How many hidden values am I allowed to observe in each round?", "How is your bonus determined?"].concat((PARAMS.PR_type !== "none" ? ["What does the feedback teach you?"] : [])),
    options: [['True', 'False'], ['$0.01', '$0.05', '$1.60', '$2.80'], ['At most 1', 'At most 5', 'At most 10', 'At most 15', 'As many or as few as I wish'], ['10% of my best score on any round', '10% of my total score on all rounds', '5% of my best score on any round', '5% of my score on a random round'], ['Whether I observed the rewards of relevant locations.', 'Whether I chose the move that was best according to the information I had.', 'The length of the delay is based on how much more money I could have earned by planning and deciding better.', 'All of the above.']],
    required: [true, true, true, true, true],
    correct: ['True', fmtMoney(PARAMS.info_cost), 'As many or as few as I wish', '5% of my score on a random round', 'All of the above.'],
    on_mistake: function(data) {
      return alert("You got at least one question wrong. We'll send you back to the\ninstructions and then you can try again.");
    }
  });
  instruct_loop = new Block({
    timeline: [instructions, quiz],
    loop_function: function(data) {
      var c, l, len3, ref2;
      ref2 = data[1].correct;
      for (l = 0, len3 = ref2.length; l < len3; l++) {
        c = ref2[l];
        if (!c) {
          return true;
        }
      }
      psiturk.finishInstructions();
      psiturk.saveData();
      return false;
    }
  });
  main = new MDPBlock({
    timeline: _.shuffle(TRIALS)
  });
  finish = new Block({
    type: 'button-response',
    stimulus: function() {
      return markdown("# You've completed the HIT\n\nThanks again for participating. We hope you had fun!\n\nBased on your performance, you will be\nawarded a bonus of **$" + (calculateBonus().toFixed(2)) + "**.");
    },
    is_html: true,
    choices: ['Submit hit'],
    button_html: '<button class="btn btn-primary btn-lg">%choice%</button>'
  });
  if (DEBUG) {
    experiment_timeline = [main, finish];
  } else {
    experiment_timeline = [instruct_loop, main, finish];
  }
  BONUS = void 0;
  calculateBonus = function() {
    var data;
    if (DEBUG) {
      return 0;
    }
    if (BONUS != null) {
      return BONUS;
    }
    data = jsPsych.data.getTrialsOfType('mouselab-mdp');
    BONUS = 0.05 * Math.max(0, (_.sample(data)).score);
    psiturk.recordUnstructuredData('final_bonus', BONUS);
    return BONUS;
  };
  reprompt = null;
  save_data = function() {
    return psiturk.saveData({
      success: function() {
        console.log('Data saved to psiturk server.');
        if (reprompt != null) {
          window.clearInterval(reprompt);
        }
        return psiturk.computeBonus('compute_bonus', psiturk.completeHIT);
      },
      error: function() {
        return prompt_resubmit;
      }
    });
  };
  prompt_resubmit = function() {
    $('#jspsych-target').html("<h1>Oops!</h1>\n<p>\nSomething went wrong submitting your HIT.\nThis might happen if you lose your internet connection.\nPress the button to resubmit.\n</p>\n<button id=\"resubmit\">Resubmit</button>");
    return $('#resubmit').click(function() {
      $('#jspsych-target').html('Trying to resubmit...');
      reprompt = window.setTimeout(prompt_resubmit, 10000);
      return save_data();
    });
  };
  return jsPsych.init({
    display_element: $('#jspsych-target'),
    timeline: experiment_timeline,
    on_finish: function() {
      if (DEBUG) {
        return jsPsych.data.displayData();
      } else {
        psiturk.recordUnstructuredData('final_bonus', calculateBonus());
        return save_data();
      }
    },
    on_data_update: function(data) {
      console.log('data', data);
      return psiturk.recordTrialData(data);
    }
  });
};
