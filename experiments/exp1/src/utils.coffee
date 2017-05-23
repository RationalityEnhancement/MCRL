converter = new showdown.Converter()
markdown = (txt) -> converter.makeHtml(txt)

getTime = -> (new Date).getTime()

loadJson = (file) ->
  result = $.ajax
    dataType: 'json'
    url: file
    async: false
  return result.responseJSON

zip = (rows...) -> rows[0].map((_,c) -> rows.map((row) -> row[c]))

check = (name, val) ->
  if val is undefined
    throw new Error "#{name}is undefined"
  val

mean = (xs) ->
  (xs.reduce ((acc, x) -> acc+x)) / xs.length

checkObj = (obj, keys) ->
  if not keys?
    keys = Object.keys(obj)
  for k in keys
    if obj[k] is undefined
      console.log 'Bad Object: ', obj
      throw new Error "#{k} is undefined"
  obj

assert = (val) ->
  if not val
    throw new Error 'Assertion Error'
  val

checkWindowSize = (width, height, display) ->
  win_width = $(window).width()
  maxHeight = $(window).height()
  if $(window).width() < width or $(window).height() < height
    display.hide()
    $('#window_error').show()
  else
    $('#window_error').hide()
    display.show()


mapObject =  (obj, fn) ->
  Object.keys(obj).reduce(
    (res, key) ->
      res[key] = fn(obj[key])
      return res
    {}
  )

deepMap = (obj, fn) ->
  deepMapper = (val) ->
    if typeof val is 'object' then (deepMap val, fn) else (fn val)
  if Array.isArray(obj)
    return obj.map(deepMapper)
  if typeof obj == 'object'
    return mapObject(obj, deepMapper)
  else
    return obj

deepLoadJson = (file) ->
  replaceFileName = (f) ->
    if typeof f is 'string' and f.endsWith '.json'
      o = loadJson f
      o._json = f
      return o
    else f
  deepMap (loadJson file), replaceFileName
