/**
 * jspsych-survey-text
 * a jspsych plugin that asks the participant for a secret code
 *
 * Fred Callaway
 *
 * documentation: Nope!
 *
 */


jsPsych.plugins['secret-code'] = (function() {

  var plugin = {};

  plugin.trial = function(display_element, trial) {
    display_element.html('');
    var preamble = '<h1>Secret Code</h1>Please enter the code you received in the previous experiment.<p>';
    trial.preamble = (typeof trial.preamble == 'undefined') ? preamble : trial.preamble;

    // if any trial variables are functions
    // this evaluates the function and replaces
    // it with the output of the function
    trial = jsPsych.pluginAPI.evaluateFunctionParameters(trial);

    // show preamble text
    display_element.append($('<div>', {
      "id": 'jspsych-survey-text-preamble',
      "class": 'jspsych-survey-text-preamble'
    }));
    $('#jspsych-survey-text-preamble').html(trial.preamble);
    
    var input = $('<input/>').attr({ type: 'text', id: 'test', name: 'test'}).appendTo(display_element);
    
    var error = $('<div/>', {
       html: "<br><b>That code is incorrect. Please try again.</b>"
    }).hide().appendTo(display_element);
    
    var success = $('<div/>', {
       html: "<br><b>Thanks!</b>"
    }).hide().appendTo(display_element);

    input.focus()
    input.keypress(function(event) {
      if (event.keyCode == 13) {  // pressed enter
        if (input.val() == trial.code) {
          error.hide()
          success.show()
          jsPsych.finishTrial({})
        }
        else {
          error.show()
        }
      }
    })
  };

  return plugin;
})();
