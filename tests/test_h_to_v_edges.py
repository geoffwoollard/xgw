import numpy as np
from xgw.Hyperplane_approx import DoubleRepresentation
from xgw.h_to_v_edges import update_edges_with_new_halfplane
import pytest


def canonicalize(V, E):
    """
    V: (n, d) vertex array
    E: (m, 2) edge array of integer indices

    Returns:
        V_canon: vertices in canonical order
        E_canon: edges relabeled consistently
    """
    # Step 1: canonical vertex ordering
    order = np.lexsort(V.T[::-1])
    V_canon = V[order]

    # Step 2: inverse permutation (old index -> new index)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))

    # Step 3: relabel edges
    E_canon = inv[E]

    # Optional: make edge order canonical too
    E_canon = np.sort(E_canon, axis=1)      # sort each edge (i,j) -> (min,max)
    E_canon = E_canon[np.lexsort(E_canon.T)]  # sort edges globally

    return V_canon, E_canon

@pytest.fixture
def cube_3d():
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    E = np.array([
        [0, 1],
        [0, 2],
        [0, 4],
        [1, 3],
        [1, 5],
        [2, 3],
        [2, 6],
        [3, 7],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
    ])
    dd = DoubleRepresentation()
    dd.add_V(V)
    A, b = dd.H
    return V, E, A, b


@pytest.fixture
def cube_2d():
    V = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    E = np.array([
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
    ])
    dd = DoubleRepresentation()
    dd.add_V(V)
    A, b = dd.H
    return V, E, A, b

def test_cube_3d_keep_one_corner(cube_3d):
    '''Test plane close to origin cutting off all except one.'''
    V, E, A, b = cube_3d
    a_new = np.array([-1.0, -1.0, -1.0])
    delta = 0.1
    b_new = np.array(-delta)

    V_final, E_final, A_final, b_final = update_edges_with_new_halfplane(V, E, A, b, a_new, b_new)

    V_final, E_final = canonicalize(V_final, E_final)
    E_true = np.array([[2, 3],
                [1, 2],
                [1, 3],
                [1, 0],
                [2, 0],
                [3, 0]])
    V_true = np.array([[0.0, 0.0, 0.0],
                [delta, 0.0, 0.0],
                [0.0, delta, 0.0],
                [0.0, 0.0, delta]])
    V_true, E_true = canonicalize(V_true, E_true)
    assert np.allclose(V_final, V_true)
    assert np.array_equal(E_final, E_true)


# def test_cube_2d_keep_one_corner(cube_2d):
#     '''Test plane close to origin cutting off all except one.'''
#     V, E, A, b = cube_2d
#     a_new = np.array([-1.0, -1.0])
#     delta = 0.1
#     b_new = np.array(-delta)

#     V_final, E_final, A_final, b_final = update_edges_with_new_halfplane(V, E, A, b, a_new, b_new)

#     V_final, E_final = canonicalize(V_final, E_final)
#     E_true = np.array([[1, 2],
#                 [0, 1],
#                 [0, 2]])
#     V_true = np.array([[0.0, 0.0],
#                 [delta, 0.0],
#                 [0.0, delta]])
#     V_true, E_true = canonicalize(V_true, E_true)
#     assert np.allclose(V_final, V_true)
#     assert np.array_equal(E_final, E_true)


def test_cube_3d_drop_one_corner(cube_3d):
    '''Test plane close to corner, cutting off one corner.'''
    V, E, A, b = cube_3d

    a_new = np.array([-1.0, -1.0, -1.0])
    max_L1_distance_to_diag_corner = 3
    delta = 0.1
    b_new = np.array(-max_L1_distance_to_diag_corner + delta)

    V_final, E_final, A_final, b_final = update_edges_with_new_halfplane(V, E, A, b, a_new, b_new)
    # print("V_final:", V_final)
    # print("E_final:", E_final)

    V_final, E_final = canonicalize(V_final, E_final)
    V_true = np.array([
        [0. , 0. , 0. ],
        [1. , 0. , 0. ],
        [0. , 1. , 0. ],
        [1. , 1. , 0. ],
        [0. , 0. , 1. ],
        [1. , 0. , 1. ],
        [0. , 1. , 1. ],
        [1. , 1. , 0.9],
        [1. , 0.9, 1. ],
        [0.9, 1. , 1. ]
    ])

    E_true = np.array([
        [0, 1],
        [0, 2],
        [0, 4],
        [1, 3],
        [1, 5],
        [2, 3],
        [2, 6],
        [4, 5],
        [4, 6],
        [7, 8],
        [8, 9],
        [7, 9],
        [7, 3],
        [8, 5],
        [9, 6]
    ], dtype=int)
    V_true, E_true = canonicalize(V_true, E_true)
    assert np.allclose(V_final, V_true)
    assert np.array_equal(E_final, E_true)

