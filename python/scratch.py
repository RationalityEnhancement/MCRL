def search(mdp, state):
    start = Node(1, state)
    frontier = PriorityQueue(key=get('f'))
    frontier.push(start)
    # reward_to_state = defaultdict(lambda: -np.inf)
    # reward_to_state[start.state] = 0
    best_finished = start

    def expand(node):
        nonlocal best_finished
        best_finished = min((best_finished, node), key=get('f'))
        for a in mdp.actions(node):
            for p, r, s in mdp.result(state, a):
                pass

        # s0, p0, r0, _ = node
        for a, s1, r, done in self.model.options(s0):
            node1 = Node(s1, p0 + [a], r0 + r, done)
            # if node1.reward <= reward_to_state[s1]:
                # continue  # cannot be better than an existing node
            self.node_history.append(
                {'plan': node1.plan,
                 'r': node1.reward,
                 'b': self.env._observe(node1.state)[-1]    ,
                 'v': -eval_node(node1)})
            # reward_to_state[s1] = node1.reward
            if done:
                best_finished = min((best_finished, node1), key=eval_node)
            else:
                frontier.push(node1)
                
    for i in range(expansions):
        if frontier:
            expand(frontier.pop())
        else:
            break



    if frontier:
        plan = min(best_finished, frontier.pop(), key=eval_node)
    else:
        plan = best_finished
    # choices = concat([completed, map(get(1), take(100, frontier))])
    # plan = min(choices, key=eval_node(noisy=True))
    self.log(
        i,
        len(plan.plan), 
        -round(eval_node(plan, noisy=False), 2),
        plan.done,
    )
    # self._trace['plans'].append(plan.plan)
    return plan.plan


import heapq
class PriorityQueue(list):
    def __init__(self, key):
        self.key = key

    def pop(self):
        return heapq.heappop(self)[1]
        
    def push(self, item):
        heapq.heappush(self, (self.key(item), item))


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, p, state, parent=None, action=None, g=0, h=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.p = p
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g
        self.h = h
        self.f = g + h
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def solution(self):
        "Return the sequence of actions to go from the root to this node."
        return [node.action for node in self.path()[1:]]

    def path(self):
        "Return a list of nodes forming the path from the root to this node."
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

def path_value(state, path):
    def rec(i, s, v, p):
        if i == len(path):
            yield p, v
            return
        s = self.tree[s][path[i]]
        r = state[s]
        if r is self.unknown:
            for o in self.outcomes:
                yield from rec(i+1, s, v+o, p * self.outcome_prob)
        else:
            yield from rec(i+1, s, v+r, p)
    return expect(rec(0, 0, 0, 1))

return max(path_value(state, path) for path in self.paths)
