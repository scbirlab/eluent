"""Implementation of Union-Find Tree with numpy."""

import numpy as np

class NumpyDisjointSet:
    """A Union-Find (Disjoint-Set) for elements 0..N-1.
    
    With path-halving find and union-by-size.

    Examples
    ========
    >>> import numpy as np; np.set_printoptions(legacy='1.25')
    >>> ds = NumpyDisjointSet(5)
    >>> ds.find(0)
    0
    >>> ds.merge(0, 1)
    >>> ds.find(1) == ds.find(0)
    True
    >>> ds.merge(2, 3)
    >>> ds.find(2) == ds.find(4)
    False
    >>> ds[3] == ds[2]
    True

    """
    def __init__(self, n: int, memmap: bool = False, cache: str = "parent.npy"):
        self.size = np.ones(n, dtype=np.int32)
        self.memmap = memmap
        if self.memmap:
            self.parent = np.memmap(
                cache,
                dtype=np.int32, 
                mode="w+", 
                shape=(n,),
            )
            self.parent[:] = np.arange(n, dtype=np.int32)
        else:
            self.parent = np.arange(n, dtype=np.int32)

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def merge(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri == rj:
            return
        if self.size[ri] < self.size[rj]:
            ri, rj = rj, ri
        self.parent[rj] = ri
        self.size[ri] += self.size[rj]

    def __getitem__(self, i: int) -> int:
        return self.find(i)
