
_ = require 'underscore'
hash = require 'object-hash'

f = (x, y) ->
  console.log x, y
  
hasher = (args) ->
  return hash(Object.values arguments)

f = _.memoize(f, hasher)


a = foo: 1
b = bar: 1

f(a, a)
f(a, a)
f(a, b)
f(b, a)
f(b, )


