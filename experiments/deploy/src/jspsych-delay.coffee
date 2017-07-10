
jsPsych.plugins['delay'] = do ->
  plugin =
    trial: (display_element, trial_config) ->
      do display_element.empty
      trial_config = jsPsych.pluginAPI.evaluateFunctionParameters trial_config

      {duration} = trial_config

      display_element.append markdown """
        # Break

        You were randomly chosen to take a #{duration} minute break.

        Feel free to do whatever you'd like until the timer completes.
        But, please try to resume the HIT as soon as possible after the
        timer is finished.

        If you're not sure what to do, you could watch a
        [cat video](https://www.youtube.com/watch?v=IytNBm8WA1c&index=1&list=PL8B03F998924DA45B&t=2s) 
        on youtube.
      """
      $timer = $('<div>', class: 'timer').appendTo display_element

      start = do getTime
      seconds = 0
      minutes = duration % 60
      hours = duration // 60

      complete = ->
        # Continue by clicking button
        $container = $('<div>', class: 'center-content').appendTo display_element
        $container.append ($('<button>')
          .addClass('btn btn-primary btn-lg')
          .text('Continue')
          .click (->
            do display_element.empty
            jsPsych.finishTrial {rt: (do getTime) - start}
          )
        )

      # TODO: This should be done with setInterval
      tick = ->
        seconds--
        if seconds < 0
          seconds = 59
          minutes--
          if minutes < 0
            minutes = 59
            hours--
            if hours < 0
              do complete
              return
        console.log 'here'
        $timer.html (if hours then (if hours > 9 then hours else '0' + hours) else '00') + ':' + (if minutes then (if minutes > 9 then minutes else '0' + minutes) else '00') + ':' + (if seconds > 9 then seconds else '0' + seconds)
        setTimeout(tick, 1000)
      do tick



  return plugin