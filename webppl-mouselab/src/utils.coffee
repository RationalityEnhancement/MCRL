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

  buildEnv: () ->
    getMoves = (addr) ->
      last = parseInt (_.last addr)
      # '0123456789'.slice(0, branch[addr.length])
      switch addr.length
        when 0
          [0, 1, 2, 3]  # up, right, down, left
        when 1
          [last]
        when 2
          [(last + 1) % 4, (last + 3) % 4]
        when 3
          []
        else
          throw new Error ('too long ' + addr)

    newLoc = (loc, move) ->
      [x, y] = loc
      options = [
        [x, y+1]
        [x+1, y]
        [x, y-1]
        [x-1, y]
      ]
      options[move]

    belief = (addr) ->
      if addr == '' then 0 else UNKNOWN
    
    tree = []
    transition = []
    initialState = []
    idx2address = []
    address2idx = {}
    layout = []

    addNode = (addr, loc) ->
      address2idx[addr] = idx = tree.length
      idx2address.push addr
      layout.push loc
      children = []
      tree.push children
      moves = {}
      transition.push moves
      initialState.push (belief addr)
      for m in (getMoves addr)
        child = addNode (addr + m), (newLoc loc, m)
        children.push child
        moves[m] = child

      return idx

    addNode('', [0, 0])

    return {
      initialState
      transition
      tree
      layout
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
