import numpy as np

def random_cut(V, rng=None):
    """
    V : (n, d) array of vertices
    Returns: (x, r) where x is a unit normal and r defines the hyperplane <v,x>=r
    """
    if rng is None:
        rng = np.random.default_rng()

    n, d = V.shape

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

def test_random_cut_and_split():
    # Example: cube in 3D
    V = np.array([
        [0,0,0],[1,0,0],[0,1,0],[1,1,0],
        [0,0,1],[1,0,1],[0,1,1],[1,1,1]
    ])

    x, r = random_cut(V)
    assert isinstance(r, float)
    assert x.shape == (3,)
    print("normal:", x)
    print("offset:", r)
    left, right, on = split_by_cut(V, x, r)
    print("left:", left)
    print("right:", right)
    print("on:", on)
    assert len(left) + len(right) + len(on) == len(V)
    assert np.allclose(np.unique(np.concatenate([left, right, on]), axis=0), np.unique(V, axis=0))

if __name__ == "__main__":
    test_random_cut_and_split()