// Generated by CoffeeScript 1.12.6

/*
experiment.coffee
Fred Callaway

Demonstrates the jsych-mdp plugin
 */
var N_TEST, N_TRAIN, N_TRIALS, SCORE, TEST_TRIALS, TRAIN_TRIALS, calculateBonus, createStartButton, delay, initializeExperiment, isIE, psiturk,
  extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
  hasProp = {}.hasOwnProperty;

psiturk = new PsiTurk(uniqueId, adServerLoc, mode);

isIE = false || !!document.documentMode;

TEST_TRIALS = void 0;

TRAIN_TRIALS = void 0;

N_TEST = 6;

N_TRAIN = 10;

N_TRIALS = 16;

SCORE = 0;

calculateBonus = void 0;

delay = function(time, func) {
  return setTimeout(func, time);
};

if (isIE) {
  $('#jspsych-target').hide();
  $('#IE_error').show();
}

$(window).on('load', function() {
  var loadTimeout, slowLoad;
  slowLoad = function() {
    return $('#failLoad').show();
  };
  loadTimeout = delay(12000, slowLoad);
  psiturk.preloadImages(['static/images/example1.png', 'static/images/example2.png', 'static/images/example3.png', 'static/images/money.png', 'static/images/plane.png', 'static/images/spider.png']);
  return delay(300, function() {
    var ERROR, condition_nr, expData, trials;
    if (SHOW_PARTICIPANT_DATA) {
      expData = loadJson("static/json/data/1B.0/stimuli/" + COST_LEVEL + "_cost.json");
    } else {
      expData = loadJson("static/json/" + COST_LEVEL + "_cost.json");
    }
    condition_nr = condition % nrConditions;
    trials = expData.blocks.standard;
    TRAIN_TRIALS = trials.slice(0, N_TRAIN);
    TEST_TRIALS = trials.slice(N_TRAIN);
    if (!SHOW_PARTICIPANT_DATA) {
      TRAIN_TRIALS = _.shuffle(TRAIN_TRIALS);
      TEST_TRIALS = _.shuffle(TEST_TRIALS);
    }
    N_TRIALS = N_TRAIN + N_TEST;
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
  var Block, MDPBlock, QuizLoop, TextBlock, debug_slide, experiment_timeline, finish, instruct_loop, instructions, msgType, prompt_resubmit, quiz, reprompt, save_data, test, text, train;
  console.log('INITIALIZE EXPERIMENT');
  msgType = (function() {
    switch (PARAMS.message) {
      case 'none':
        return '_noMsg';
      case 'simple':
        return '_simpleMsg';
      default:
        return '';
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
        if (PARAMS.PR_type === "objectLevel") {
          return [markdown("# Instructions\n\n<b>You will receive feedback about your planning. This feedback\nwill help you learn how to make better decisions.</b> After each\nflight, if you did not make the best move, a feedback message\nwill apear. This message will tell you whether you flew along\nthe best route given your current location, and what the best\nmove would have been.\n\nThis feedback will be presented after each of the first\n" + N_TRAIN + " rounds; during the final " + N_TEST + " rounds,\nno feedback will be presented.\n\nIn the example below, the best move was not taken. As a result,\nthere is a 15 second timeout penalty.<b> The duration of the\ntimeout penalty is proportional to how poor of a move you made:\n</b> the more money you could have earned, the longer the delay.\n<b> If you perform optimally, no feedback will be shown and you\ncan proceed immediately.</b> \n\n" + (img('task_images/Slide5.png')) + "\n")];
        } else if (PARAMS.PR_type === "demonstration") {
          return [markdown("# Instructions\n\n<b>You will receive guidance about how to plan. This guidance\nwill help you learn how to make better decisions.</b> In the\nfirst " + N_TRAIN + " rounds, an expert will demonstrate what optimal\nplanning and flight paths look like. In the remaining " + N_TEST + "\nrounds, you will make your own choices.")];
        } else if (PARAMS.message === "simple") {
          return [markdown("# Instructions\n\n<b>You will receive feedback about your planning. This feedback will\nhelp you learn how to make better decisions.</b> After each flight, if\nyou did not plan optimally, a feedback message will apear.\n\nIn the example below, there is a 26 second timeout penalty. <b>The duration of the timeout penalty is\nproportional to how poorly you planned your route:</b> the more\nmoney you could have earned from observing more/less values\nand/or choosing a better route, the longer the delay. <b>If you\nperform optimally, no feedback will be shown and you can proceed\nimmediately.</b> The example message here is not necessarily\nrepresentative of the feedback you'll receive.\n\nThis feedback will be presented after each of the first\n" + N_TRAIN + " rounds; during the final " + N_TEST + " rounds,\nno feedback will be presented.\n\n" + (img('task_images/Slide4_simple.png')))];
        } else {
          return [markdown("# Instructions\n\n<b>You will receive feedback about your planning. This feedback will\nhelp you learn how to make better decisions.</b> After each flight, if\nyou did not plan optimally, a feedback message will apear. This message\nwill tell you two things:\n\n1. Whether you observed too few relevant values or if you observed\n   irrelevant values (values of locations that you can't fly to).\n2. Whether you flew along the best route given your current location and\n   the information you had about the values of other locations.\n\nThis feedback will be presented after each of the first\n" + N_TRAIN + " rounds; during the final " + N_TEST + " rounds,\nno feedback will be presented.\n\nIn the example below, there is a 6 second timeout penalty. If\nyou observed too few relevant values, the message would say,\n\"You should have gathered more information!\"; if you observed\ntoo many values, it would say \"You should have gathered less\ninformation!\". <b>The duration of the timeout penalty is\nproportional to how poorly you planned your route:</b> the more\nmoney you could have earned from observing more/less values\nand/or choosing a better route, the longer the delay. <b>If you\nperform optimally, no feedback will be shown and you can proceed\nimmediately.</b> The example message here is not necessarily\nrepresentative of the feedback you'll receive.\n\n" + (img('task_images/Slide4_neutral2.png')))];
        }
      } else if (PARAMS.message === "full") {
        return [markdown("# Instructions\n\n<b>You will receive feedback about your planning. This feedback will\nhelp you learn how to make better decisions.</b> After each flight a feedback message will apear. This message\nwill tell you two things:\n\n1. Whether you observed too few relevant values or if you observed\n   irrelevant values (values of locations that you can't fly to).\n2. Whether you flew along the best route given your current location and\n   the information you had about the values of other locations.\n\nThis feedback will be presented after each of the first\n" + N_TRAIN + " rounds; during the final " + N_TEST + " rounds,\nno feedback will be presented.\n\nIf you observe too few relevant values, the message will say,\n\"You should have gathered more information!\"; if you observe\ntoo many values, it will say \"You should have gathered less\ninformation!\"; and the image below shows the message you will see when you collected the right information but used it incorrectly.\n\n" + (img('task_images/Slide4_neutral.png')))];
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
      var c, i, len, ref;
      console.log('data', data);
      ref = data[data.length].correct;
      for (i = 0, len = ref.length; i < len; i++) {
        c = ref[i];
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
    pages: [markdown("# Instructions " + (text.debug()) + "\n\nIn this game, you are in charge of flying an aircraft. As shown below,\nyou will begin in the central location. The arrows show which actions\nare available in each location. Note that once you have made a move you\ncannot go back; you can only move forward along the arrows. There are\neight possible final destinations labelled 1-8 in the image below. On\nyour way there, you will visit two intermediate locations. <b>Every\nlocation you visit will add or subtract money to your account</b>, and\nyour task is to earn as much money as possible. <b>To find out how much\nmoney you earn or lose in a location, you have to click on it.</b> You\ncan uncover the value of as many or as few locations as you wish.\n\n" + (img('task_images/Slide1.png')) + "\n\nTo navigate the airplane, use the arrows (the example above is non-interactive).\nYou can uncover the value of a location at any time. Click \"Next\" to proceed."), markdown("# Instructions\n\nYou will play the game for " + N_TRIALS + " rounds. The value of\nevery location will change from each round to the next. At the\nbegining of each round, the value of every location will be hidden,\nand you will only discover the value of the locations you click on.\nThe example below shows the value of every location, just to give you\nan example of values you could see if you clicked on every location.\n<b>Every time you click a circle to observe its value, you pay a fee\nof " + (fmtMoney(PARAMS.info_cost)) + ".</b>\n\n" + (img('task_images/Slide2_' + COST_LEVEL + '.png')) + "\n\nEach time you move to a\nlocation, your profit will be adjusted. If you move to a location with\na hidden value, your profit will still be adjusted according to the\nvalue of that location. " + (text.constantDelay()))].concat((text.feedback()).concat([markdown("# Instructions\n\nThere are two more important things to understand:\n1. You must spend at least 45 seconds on each round. A countdown timer\n   will show you how much more time you must spend on the round. You\n   won’t be able to proceed to the next round before the countdown has\n   finished, but you can take as much time as you like afterwards.\n2. </b>You will earn <u>real money</u> for your flights.</b>\n   Specifically, for every $10 you earn in the game, we will add 5 cents to your bonus. Please note that each and every one of the\n   " + N_TRIALS + " rounds counts towards your bonus.\n\n" + (img('task_images/Slide3.png')) + "\n\n You may proceed to take an entry quiz, or go back to review the instructions.")])),
    show_clickable_nav: true
  });
  quiz = new Block({
    preamble: function() {
      return markdown("# Quiz");
    },
    type: 'survey-multi-choice',
    questions: ["True or false: The hidden values will change each time I start a new round.", "How much does it cost to observe each hidden value?", "How many hidden values am I allowed to observe in each round?", "How is your bonus determined?"].concat((PARAMS.PR_type !== "none" & PARAMS.PR_type !== "demonstration" ? ["What does the feedback teach you?"] : [])),
    options: [['True', 'False'], ['$0.01', '$0.05', '$1.00', '$2.50'], ['At most 1', 'At most 5', 'At most 10', 'At most 15', 'As many or as few as I wish'], ['1% of my best score on any round', '5 cents for every $10 I earn in each round', '10% of my best score on any round', '10% of my score on a random round']].concat((PARAMS.PR_type === "objectLevel" ? [['Whether I chose the move that was best.', 'The length of the delay is based on how much more money I could have earned.', 'All of the above.']] : PARAMS.PR_type !== "none" ? [['Whether I observed the rewards of relevant locations.', 'Whether I chose the move that was best according to the information I had.', 'The length of the delay is based on how much more money I could have earned by planning and deciding better.', 'All of the above.']] : [])),
    required: [true, true, true, true, true],
    correct: ['True', fmtMoney(PARAMS.info_cost), 'As many or as few as I wish', '5 cents for every $10 I earn in each round', 'All of the above.'],
    on_mistake: function(data) {
      return alert("You got at least one question wrong. We'll send you back to the\ninstructions and then you can try again.");
    }
  });
  instruct_loop = new Block({
    timeline: [instructions, quiz],
    loop_function: function(data) {
      var c, i, len, ref;
      ref = data[1].correct;
      for (i = 0, len = ref.length; i < len; i++) {
        c = ref[i];
        if (!c) {
          return true;
        }
      }
      psiturk.finishInstructions();
      psiturk.saveData();
      return false;
    }
  });
  train = new MDPBlock({
    demonstrate: PARAMS.PR_type === "demonstration",
    timeline: TRAIN_TRIALS
  });
  test = new Block({
    timeline: (function() {
      var tl;
      tl = [];
      if (PARAMS.feedback) {
        tl.push(new TextBlock({
          text: markdown("# No more feedback\n\nYou are now entering a block without feedback. There will be no\nmessages and no delays regardless of what you do, but your\nperformance still affects your bonus.\n\nPress **space** to continue.")
        }));
      }
      if (PARAMS.PR_type === "demonstration") {
        tl.push(new TextBlock({
          text: markdown("# Your turn\n\nThis was the last demonstration from your teacher. Now it is your\nturn to decide which locations to inspect and where to fly to.\n\nPress **space** to continue.")
        }));
      }
      tl.push(new MDPBlock({
        feedback: false,
        timeline: TEST_TRIALS
      }));
      return tl;
    })()
  });
  console.log('test', test);
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
    experiment_timeline = [instruct_loop, train, test, finish];
  } else {
    experiment_timeline = [instruct_loop, train, test, finish];
  }
  calculateBonus = function(final) {
    var bonus, data;
    if (final == null) {
      final = false;
    }
    data = jsPsych.data.getTrialsOfType('mouselab-mdp');
    bonus = (Math.max(0, SCORE)) * PARAMS.bonus_rate;
    bonus = (Math.round(bonus * 100)) / 100;
    return bonus;
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
