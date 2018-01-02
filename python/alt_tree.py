from toolz import memoize, get, sliding_window, concatv
from tree import Tree

class TreeStructure(list):
    def __init__(self, *args):
        super().__init__(*args)
        

    def hash(self):
        return id(self)

    def __setitem__(self):
        assert 0

    def __delitem__(self):
        assert 0

    def subtree(self, idx):
        children = []
        child_indices = self.structure[0]
        for idx, stop in sliding_window(2, concatv(child_indices, [None])):
            structure = tuple(tuple(i - idx for i in indices)
                              for indices in self.structure[idx:stop])
            children.append(self.new(self.values[idx:stop], structure))
        return children


    @property
    # @memoize
    def subtrees(self):
        slices = [0] * len(self)
        def get_end(n):
            end = max((get_end(n1) for n1 in self[n]), default=n+1)
            slices[n] = slice(n, end)
            return end
        get_end(0)
        return slices

    @property
    @memoize
    def paths(self):
        def rec(path):
            children = self[path[-1]]
            if children:
                for child in children:
                    yield from rec(path + [child])
            else:
                yield path
        return list(rec([self]))


    @memoize
    def path_to(self, idx):
        path = [0]
        if idx == 0:
            return path
        for _ in range(10000):
            children = self[path[-1]]
            for i, child in enumerate(children):
                if child == idx:
                    path.append(idx)
                    return path
                if child > idx:
                    path.append(children[i-1])
                    break
            else:
                path.append(child)
        assert False

    @classmethod
    @memoize
    def build(cls, branching):
        # if branching:
        #     return TreeStructure(cls.build(branching[1:])
        #                          for _ in range(branching[0]))
        # else:
        #     return TreeStructure()

        structure = []
        return

        def expand(d):
            my_idx = len(vals)
            vals.append(value(d))
            children = []
            structure.append(children)
            for _ in range(get(d, branching, 0)):
                child_idx = expand(d+1)
                children.append(child_idx)
            return my_idx

        expand(0)
        return cls(tuple(vals), TreeStructure(structure))


      

class FlatTree(Tree):
    """Flat representation of a tree"""

    def __init__(self, values, structure):
        self.values = values
        self.structure = structure
        self.val = values[0]

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, idx):
        return self.values[idx]

    @staticmethod
    @memoize
    def new(values, structure):
        return FlatTree(values, structure)

    @property
    @memoize
    def children(self):
        children = []
        child_indices = self.structure[0]
        for idx, stop in sliding_window(2, concatv(child_indices, [None])):
            structure = tuple(tuple(i - idx for i in indices)
                              for indices in self.structure[idx:stop])
            children.append(self.new(self.values[idx:stop], structure))
        return children

    def update(self, idx, val):
        values = list(self.values)
        values[idx] = val
        return FlatTree(tuple(values), self.structure)

    def path_to(self, idx):
        return [self.values[i] for i in self.structure.path_to(idx)]

    def paths(self):
        return [self.values[i] for i in self.structure.paths()]
    
    def subtree(self, idx):
        subtree = self.structure.subtrees[idx]
        return self.new(self.values[subtree], self.structure[subtree])


    @classmethod
    def build(cls, branching, value):
        if not callable(value):
            val = value
            value = lambda depth: val

        vals = []
        structure = []

        def expand(d):
            my_idx = len(vals)
            vals.append(value(d))
            children = []
            structure.append(children)
            for _ in range(get(d, branching, 0)):
                child_idx = expand(d+1)
                children.append(child_idx)
            return my_idx

        expand(0)
        return cls(tuple(vals), TreeStructure(structure))

