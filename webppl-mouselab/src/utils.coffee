_ = require 'lodash'
LRU = require 'lru-cache'
TERM_ACTION = '__TERM_ACTION__'
TERM_STATE = '__TERM_STATE__'
UNKNOWN = '__'
INITIAL_NODE = ''


module.exports = {
  
  update: (obj, key, val) ->
    o = {}
    o[key] = val
    _.extend({}, obj, o)

  updateList: (lst, idx, val) ->
    lst = lst.slice()
    lst[idx] = val
    return lst

  buildEnv: (branch) ->
    branch = branch.concat [0]  # leaves have 0 children

    actions = (addr) ->
      '0123456789'.slice(0, branch[addr.length])
    belief = (addr) ->
      if addr == '' then 0 else UNKNOWN
    
    tree = []
    initialState = []
    idx2address = []
    address2idx = {}
    layout = []

    addNode = (addr) ->
      address2idx[addr] = idx = tree.length
      idx2address.push addr
      children = []
      tree.push children
      initialState.push (belief addr)
      for a in (actions addr)
        children.push (addNode (addr + a))
      return idx

    addNode('')

    return {
      initialState
      tree
      idx2address
      address2idx
      hashState: (state) -> state
    }

  buildCross: (branch, depth) ->
    env = @buildEnv([branch] .concat Array(depth-1).fill(1))
    
    hashCache = LRU(100000)
    env.hashState = (state) ->
      if state is TERM_STATE
        state
      else
        s = state
        [[s['0'], s['00'], s['000']].sort()
         [s['1'], s['10'], s['100']].sort()
         [s['2'], s['20'], s['200']].sort()
         ].sort()
    env


  firstUnobserved: (acts) ->
    depth = _.max((a.length for a in acts))
    result = []
    for branch in '0123456789'
      for i in [0...depth]
        n = branch + ('0'.repeat i)
        if n in acts
          result.push n
          break
    return result

  cache: (f, env, maxSize=null) ->
    c = LRU(maxSize)
    usage = 0

    cf = (s, k, a) ->
      args = Array::slice.call(arguments, 3)
      stringedArgs = JSON.stringify([env.hashState args[0]] .concat args.slice(1))
      if c.has(stringedArgs)
        usage += 1
        # if usage % 50 is 0
        #   console.log 'cache', usage
        k s, c.get(stringedArgs)
      else

        newk = (s, r) ->
          if c.has(stringedArgs)
            # This can happen when cache is used on recursive functions
            console.log 'Already in cache:', stringedArgs
            if JSON.stringify(c.get(stringedArgs)) != JSON.stringify(r)
              console.log 'OLD AND NEW CACHE VALUE DIFFER!'
              console.log 'Old value:', c.get(stringedArgs)
              console.log 'New value:', r
          c.set stringedArgs, r
          k s, r

        f.apply this, [
          s
          newk
          a
        ].concat(args)

    # make the cache publically available to facillitate checking the complexity of algorithms
    cf.cache = c
    cf

  cacheFred: (f, maxSize=null) ->
    console.log 'cacheFred'
    c = LRU(maxSize)
    usage = 0

    cf = (s, k, a) ->
      args = Array::slice.call(arguments, 3)
      stringedArgs = JSON.stringify(args)
      if c.has(stringedArgs)
        usage += 1
        # if usage % 50 is 0
        #   console.log 'cache', usage
        k s, c.get(stringedArgs)
      else

        newk = (s, r) ->
          if c.has(stringedArgs)
            # This can happen when cache is used on recursive functions
            console.log 'Already in cache:', stringedArgs
            if JSON.stringify(c.get(stringedArgs)) != JSON.stringify(r)
              console.log 'OLD AND NEW CACHE VALUE DIFFER!'
              console.log 'Old value:', c.get(stringedArgs)
              console.log 'New value:', r
          c.set stringedArgs, r
          k s, r

        f.apply this, [
          s
          newk
          a
        ].concat(args)
    # make the cache publically available to facillitate checking the complexity of algorithms
    cf.cache = c
    cf

  

}


  

# e = module.exports
# console.log(
#   e.buildCross()
# )
