from toolz import memoize, last


class Tree(object):
    def __init__(self, val, children):
        self.val = val
        self.children = children

    def __getitem__(self, address):
        return last(self.path(address))

    def __iter__(self):
        yield self.val
        for child in self.children:
            yield from child


    @memoize
    def __len__(self):
        return 1 + sum(len(c) for c in self.children)

    def __hash__(self):
        return id(self)
        # return hash(self.vals) + hash(self.structure)

    def __repr__(self):
        return f'Tree({self.val})'

    def __str__(self):
        children = tuple(str(c) for c in self.children)
        return f'({self.val}, {children})'

    def update(self, address, val):
        if address:
            children = list(self.children)
            first, rest = address[0], address[1:]
            children[first] = children[first].update(rest, val)
            return Tree(self.val, children)
        else:
            return Tree(val, self.children)

    def path(self, address):
        t = self
        yield t
        for i in address:
            t = t.children[i]
            yield t

    @memoize
    def paths(self):
        def rec(path):
            if path[-1].children:
                for child in path[-1].children:
                    yield from rec(path + [child])
            else:
                yield path

        return rec([self])
    
    def value(self, max=max):
        return max(child.value(max) + child.val
                   for child in self.children)

    @classmethod
    def build(cls, branching, value):
        if not callable(value):
            val = value
            value = lambda depth: val

        def rec(d):
            v = value(d)
            if d == len(branching):
                children = []
            else:
                children = [rec(d+1) for _ in range(branching[d])]
            return Tree(v, children)

        return rec(0)

    def as_tuple(self):
        return (self.val,
                tuple(child.as_tuple() for child in self.children))

    def draw(self):
        from graphviz import Digraph
        from IPython.display import display
        
        def color(val):
            if val > 0:
                return '#8EBF87'
            else:
                return '#F7BDC4'
        
        dot = Digraph()
        def rec(tree):
            dot.node(str(id(tree)), label=repr(tree.val))
            for child in tree.children:
                dot.edge(str(id(tree)), str(id(child)))
                rec(child)

        rec(self)
        display(dot)


class TreeIndex(list):
    """A flat index over nodes of a tree."""
    def __init__(self, tree):
        
        def rec(path, tree):
            yield path
            if tree.children:
                for i, child in enumerate(tree.children):
                    yield from rec(path + (i,), child)

        super().__init__(rec((), tree))



