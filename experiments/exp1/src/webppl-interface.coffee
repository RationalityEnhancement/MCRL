
# These are all the functions available to the 
# callWebppl interface. This could be defined
# in a less hacky way.
FUNCTIONS = """
{
  initialize() {
    null
  },
  flip() {
    flip()
  },
  sampleMean(param) {
    expectation(Infer({model() {
      gaussian(param.mu, param.sigma)
    }, method:'forward', samples:param.nSample}))
  },
  PR(arg) {
    calculatePR(arg)
  }
}
"""

# Returns a function callWebppl that can be used to
# interact with a constantly running webppl process.
# callWebppl has the signature (funcName, arg) -> Promise
# where funcName must be a key in FUNCTIONS above and
# the Promise resolves to FUNCTIONS[funcName](arg)
startWebppl = () ->

  # We use promises to represent the results of callWebppl as well as the
  # callback that webppl will provide when that computation is finished.
  # We flout best practices by making the resolve function external, allowing
  # the promise to be resolved by the callback _return2js
  resolveResult = undefined
  resultPromise = new Promise (resolve) ->
    resolveResult = resolve
  resolveWebppl = undefined
  webpplPromise = new Promise (resolve) ->
    resolveWebppl = resolve

  callWebppl = (kind, arg) ->
    # We have to wait until webppl is available to make another request.
    # Because promises can be chained, you can call callWebppl many times
    # before the first result is resolved. The queue of requests is
    # represented by a chain of webpplPromises.
    webpplPromise.then (runWebppl) ->

      resultPromise = new Promise (resolve) ->
        resolveResult = resolve
      webpplPromise = new Promise (resolve) ->
        resolveWebppl = resolve

      runWebppl [kind, arg]
      return resultPromise
    
  store =
    _return2js: (result, callback) ->
      resolveResult result
      resolveWebppl callback

  code = """
  var loop = function(request) {
    var _functions = #{FUNCTIONS}
    var funcName = request[0];
    var arg = request[1];
    var func = _functions[funcName];
    loop(callAsync(globalStore._return2js, func(arg)))
  }
  loop(['initialize'])
  """

  callback = (s, val) ->
    console.log 'Terminated webppl session.'
  webppl.run code, callback, initialStore: store
  
  return callWebppl


callWebppl = startWebppl()

# Example usage
# p1 = callWebppl 'flip'
# p1.then (v) -> console.log "flip returned #{v}"
# p2 = callWebppl 'sampleMean',
#   mu: 0
#   sigma: 1
#   nSample: 1000
# p2.then (v) -> console.log "sampleMean returned #{v}"
