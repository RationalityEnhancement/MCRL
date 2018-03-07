### jspsych-webppl.js
# Fred Callaway
###

jsPsych.plugins.webppl = do ->
  plugin = {}
  console.log 'LOAD WEBPPL'

  plugin.trial = (display_element, trial) ->
    # if any trial variables are functions
    # this evaluates the function and replaces
    # it with the output of the function
    trial = jsPsych.pluginAPI.evaluateFunctionParameters(trial)
    display_element.html ''
    
    if trial.file?
      $.ajax
        url: trial.file,
        async: false
        dataType: 'text'
        success: (code) ->
          trial.code = code
      # $.get trial.file, (code) ->
      #   async: false
      #   trial.code = code
    if not trial.code?
      throw new Error('jspsych-webppl: No code or file provided!')

    if not trial.globalStore?
      trial.globalStore = {}

    trial.globalStore.display_element = display_element
    trial.globalStore.getKeyboardResponse = (params, callback) ->
      default_params =
        callback_function: callback
        valid_responses: []
        persist: false
      params = _.extend default_params, params
      console.log('params', params)
      jsPsych.pluginAPI.getKeyboardResponse params


    if not trial.onFinish?
      trial.onFinish = (s, val) -> console.log "webppl returned #{val}"

    # delay 0, ->
    #   webppl.run webppl_code, ((s, val) ->
    #     console.log 'done running'
    #     # display_element.html(JSON.stringify(val))
    #     return
    #   ), initialStore: globals
    #   return
    # return

    webppl.run trial.code, trial.onFinish, initialStore: trial.globalStore

    # display_element.click(do_thing)
    # jsPsych.pluginAPI.getKeyboardResponse
      # callback_function: do_thing
      # valid_responses: [ 'space' ]
      # persist: true
    # return

  plugin
