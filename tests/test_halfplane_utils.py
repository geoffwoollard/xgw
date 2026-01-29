import numpy as np

from xgw.halfplane_utils import random_cut, split_by_cut
from test_h_to_v_edges import cube_3d


def test_random_cut_and_split(cube_3d):
    # Example: cube in 3D
    V , _, _, _ = cube_3d

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

