import numpy as np
import pytest

from xgw.Hyperplane_approx import DoubleRepresentation
from xgw.h_to_v_edges import update_edges_with_new_halfplane, find_edges
from xgw.halfplane_utils import random_cut, split_by_cut


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

# @pytest.fixture
# def cube_4d():
#     V = np.array([
#         [0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0],
#         [0.0, 1.0, 0.0, 0.0],
#         [1.0, 1.0, 0.0, 0.0],
#         [0.0, 0.0, 1.0, 0.0],
#         [1.0, 0.0, 1.0, 0.0],
#         [0.0, 1.0, 1.0, 0.0],
#         [1.0, 1.0, 1.0, 0.0],
#         [0.0, 0.0, 0.0, 1.0],
#         [1.0, 0.0, 0.0, 1.0],
#         [0.0, 1.0, 0.0, 1.0],
#         [1.0, 1.0, 0.0, 1.0],
#         [0.0, 0.0, 1.0, 1.0],
#         [1.0, 0.0, 1.0, 1.0],
#         [0.0, 1.0, 1.0, 1.0],
#         [1.0, 1.0, 1.0, 1.0],
#     ])
#     E = np.array([
#         [0, 1],
#         [0, 2],
#         [0, 4],
#         [1, 3],
#         [1, 5],
#         [2, 3],
#         [2, 6],
#         [3, 7],
#         [4, 5],
#         [4, 6],
#         [5, 7],
#         [6, 7],
#         [0+8, 1+8],
#         [0+8, 2+8],
#         [0+8, 4+8],
#         [1+8, 3+8],
#         [1+8, 5+8],
#         [2+8, 3+8],
#         [2+8, 6+8],
#         [3+8, 7+8],
#         [4+8, 5+8],
#         [4+8, 6+8],
#         [5+8, 7+8],
#         [6+8, 7+8],
#         [0, 8],
#         [1, 9],
#         [2, 10],
#         [3, 11],
#         [4, 12],
#         [5, 13],
#         [6, 14],
#         [7, 15]
#     ])
#     dd = DoubleRepresentation()
#     dd.add_V(V)
#     A, b = dd.H
#     return V, E, A, b

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
    
    
def test_cube_3d_horizontal_cut(cube_3d):
    '''Test plane cut in half horizontally.'''
    V, E, A, b = cube_3d

    a_new = np.array([0, 0, 1.0])
    half = 1/2
    b_new = np.array(half)

    V_final, E_final, A_final, b_final = update_edges_with_new_halfplane(V, E, A, b, a_new, b_new)
    V_true = np.array([
        [0. , 0. , 1. ],
        [1. , 0. , 1. ],
        [0. , 1. , 1. ],
        [1. , 1. , 1. ],
        [0. , 0. , 0.5],
        [1. , 0. , 0.5],
        [0. , 1. , 0.5],
        [1. , 1. , 0.5]
        ])
    E_true = np.array([
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
    V_final, E_final = canonicalize(V_final, E_final)
    V_true, E_true = canonicalize(V_true, E_true)
    assert np.allclose(V_final, V_true)
    assert np.array_equal(E_final, E_true)
    
    
def test_cube_4d_drop_one_corner(cube_4d):
    '''Test plane close to corner, cutting off one corner.'''
    V, E, A, b = cube_4d

    a_new = np.array([-1.0, -1.0, -1.0, -1.0])
    max_L1_distance_to_diag_corner = 4
    delta = 0.1
    b_new = np.array(-max_L1_distance_to_diag_corner + delta)

    V_final, E_final, A_final, b_final = update_edges_with_new_halfplane(V, E, A, b, a_new, b_new)
    # print("V_final:", V_final)
    # print("E_final:", E_final)

    V_final, E_final = canonicalize(V_final, E_final)
    
    V_true = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
        [0.9, 1.0, 1.0, 1.0],
        [1.0, 0.9, 1.0, 1.0],
        [1.0, 1.0, 0.9, 1.0],
        [1.0, 1.0, 1.0, 0.9],
    ])
    
    E_true = np.array([
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
        [8, 9],
        [8, 10],
        [8, 12],
        [9, 11],
        [9, 13],
        [10, 11],
        [10, 14],
        [12, 13],
        [12, 14],
        [0, 8],
        [1, 9],
        [2, 10],
        [3, 11],
        [4, 12],
        [5, 13],
        [6, 14],
        [7, 18],
        [14, 15],
        [13, 16],
        [11, 17],
        [15, 16],
        [15, 17],
        [15, 18],
        [16, 17],
        [16, 18],
        [17, 18]
        
    ])
    
    V_true, E_true = canonicalize(V_true, E_true)
    assert np.allclose(V_final, V_true)
    assert np.array_equal(E_final, E_true)
    
    
def test_cube_4d_horizontal_cut(cube_4d):
    '''Test plane close to corner, cutting off one corner.'''
    V, E, A, b = cube_4d

    a_new = np.array([-1.0, 0, 0, 0])
    b_new = np.array(-0.5)

    V_final, E_final, A_final, b_final = update_edges_with_new_halfplane(V, E, A, b, a_new, b_new)
    # print("V_final:", V_final)
    # print("E_final:", E_final)

    V_final, E_final = canonicalize(V_final, E_final)
    
    V_true = np.array([
        [0.0, 0.0, 0.0, 0.0], # 0 
        [0.5, 0.0, 0.0, 0.0], # 1
        [0.0, 1.0, 0.0, 0.0], # 2
        [0.5, 1.0, 0.0, 0.0], # 3
        [0.0, 0.0, 1.0, 0.0], # 4
        [0.5, 0.0, 1.0, 0.0], # 5
        [0.0, 1.0, 1.0, 0.0], # 6
        [0.5, 1.0, 1.0, 0.0], # 7
        [0.0, 0.0, 0.0, 1.0], # 8
        [0.5, 0.0, 0.0, 1.0], # 9
        [0.0, 1.0, 0.0, 1.0], # 10
        [0.5, 1.0, 0.0, 1.0], # 11
        [0.0, 0.0, 1.0, 1.0], # 12
        [0.5, 0.0, 1.0, 1.0], # 13
        [0.0, 1.0, 1.0, 1.0], # 14
        [0.5, 1.0, 1.0, 1.0], # 15
    ])
    E_true = np.array([
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
        [0+8, 1+8],
        [0+8, 2+8],
        [0+8, 4+8],
        [1+8, 3+8],
        [1+8, 5+8],
        [2+8, 3+8],
        [2+8, 6+8],
        [3+8, 7+8],
        [4+8, 5+8],
        [4+8, 6+8],
        [5+8, 7+8],
        [6+8, 7+8],
        [0, 8],
        [1, 9],
        [2, 10],
        [3, 11],
        [4, 12],
        [5, 13],
        [6, 14],
        [7, 15]
    ])
    
    V_true, E_true = canonicalize(V_true, E_true)
    assert np.allclose(V_final, V_true)
    # print(E_final, len(E_final))
    # print(E_true, len(E_true))
    assert np.array_equal(E_final, E_true)

def test_find_edges(cube_2d, cube_3d, cube_4d):
    V, E, A, b = cube_2d
    V_canon, E_canon = canonicalize(V, E)
    Computed_E = find_edges(V_canon)
    Computed_E = np.sort(Computed_E, axis=1)      # sort each edge (i,j) -> (min,max)
    Computed_E = Computed_E[np.lexsort(Computed_E.T)] 
    assert np.allclose(E_canon, Computed_E)
    
    V, E, A, b = cube_3d
    V_canon, E_canon = canonicalize(V, E)
    Computed_E = find_edges(V_canon)
    Computed_E = np.sort(Computed_E, axis=1)      # sort each edge (i,j) -> (min,max)
    Computed_E = Computed_E[np.lexsort(Computed_E.T)] 
    assert np.allclose(E_canon, Computed_E)
    
    V, E, A, b = cube_4d
    V_canon, E_canon = canonicalize(V, E)
    Computed_E = find_edges(V_canon)
    Computed_E = np.sort(Computed_E, axis=1)      # sort each edge (i,j) -> (min,max)
    Computed_E = Computed_E[np.lexsort(Computed_E.T)] 
    assert np.allclose(E_canon, Computed_E)

def make_cube_any_d(d):
    V = np.array(np.meshgrid(*[[0,1]]*d)).T.reshape(-1, d)

    # Map vertex -> index for fast lookup
    index = {tuple(v): i for i, v in enumerate(V)}

    edges = []

    for i, v in enumerate(V):
        for k in range(d):
            u = v.copy()
            u[k] ^= 1   # flip one bit
            j = index[tuple(u)]

            # avoid duplicates (i,j) and (j,i)
            if i < j:
                edges.append((i, j))

    E = np.array(edges, dtype=int)

    # Your polytope stuff
    dd = DoubleRepresentation()
    dd.add_V(V)
    A, b = dd.H

    return V, E, A, b

@pytest.fixture
def cube_9d():
    return make_cube_any_d(9)

@pytest.fixture
def cube_4d():
    return make_cube_any_d(4)

def test_cube_d_random_plane_cut():
    '''Test random plane cut in cube of any dimension.'''

    n_trials = 10
    for d in [3, 4, 5]:
        V, E, A, b = make_cube_any_d(d)


        for _trial in range(n_trials):
            a_new, b_new = random_cut(V, rng=np.random.default_rng(_trial))
            left, right, on = split_by_cut(V, a_new, b_new)
            print(d, len(left), len(right), len(on))

            V_final, E_final, A_final, b_final = update_edges_with_new_halfplane(V, E, A=A, b=b, a_new=a_new, b_new=b_new)
            # Just check that some vertices remain
            assert len(V_final) > 0, 'failed for d={}, trial={}'.format(d, _trial)

if __name__ == "__main__":
    test_cube_d_random_plane_cut()