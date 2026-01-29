import numpy as np


def random_cut(V, rng=None, tol = 0.1):
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
    dist = (x_max - x_min)*tol/2
    
    

    # 3. uniform random offset in [x_min, x_max]
    r = rng.uniform(x_min+dist, x_max-dist)

    return x, r


def split_by_cut(V, x, r, atol=1e-12):
    vals = V @ x - r
    left  = V[vals < -atol]
    right = V[vals > atol]
    on    = V[np.isclose(vals, 0, atol=atol)]
    return left, right, on

