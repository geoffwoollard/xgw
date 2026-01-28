import numpy as np

def random_cut(V, rng=None):
    """
    V : (n, d) array of vertices
    Returns: (x, r) where x is a unit normal and r defines the hyperplane <v,x>=r
    """
    if rng is None:
        rng = np.random.default_rng()

    d = V.shape[1]

    # 1. random unit vector
    x = rng.normal(size=d)
    x /= np.linalg.norm(x)

    # 2. project vertices onto x
    proj = V @ x
    x_min = proj.min()
    x_max = proj.max()

    # 3. uniform random offset in [x_min, x_max]
    r = rng.uniform(x_min, x_max)

    return x, r

def split_by_cut(V, x, r):
    vals = V @ x - r
    left  = V[vals < 0]
    right = V[vals > 0]
    on    = V[np.isclose(vals, 0)]
    return left, right, on

