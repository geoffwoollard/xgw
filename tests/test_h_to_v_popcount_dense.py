import numpy as np
import pytest

from xgw.h_to_v_popcount import ExtremePointPolytope
from xgw.utils import cast_to_dense_if_sparse

def d_cube_polytope(dim, use_D_sparse):
    """
    Construct the unit hypercube [0,1]^dim
    in extreme-point + vertex-incidence form.
    """

    n_vertices = 2 ** dim
    n_facets = 2 * dim

    # --- Vertices: all binary vectors of length dim ---
    E = np.array([
        [int(b) for b in f"{i:0{dim}b}"]
        for i in range(n_vertices)
    ], dtype=float)

    # --- Incidence matrix ---
    B = np.zeros((n_facets, n_vertices), dtype=np.uint8)

    # x_i = 0 facets
    for i in range(dim):
        B[i] = (E[:, i] == 0)

    # x_i = 1 facets
    for i in range(dim):
        B[dim + i] = (E[:, i] == 1)

    return ExtremePointPolytope(E, B, use_D_sparse=use_D_sparse)

def cube_polytope(dim, use_D_sparse):
    """
    Create unit cube with correct B matrix.
    """

    if dim == 3:
        E = np.array([
            [0,0,0],
            [1,0,0],
            [0,1,0],
            [0,0,1],
            [1,1,0],
            [1,0,1],
            [0,1,1],
            [1,1,1]
        ], float)

        # constraint ordering:
        # x=0,y=0,z=0,x=1,y=1,z=1
        B = np.array([
            [1,0,1,1,0,0,1,0],
            [1,1,0,1,0,1,0,0],
            [1,1,1,0,1,0,0,0],
            [0,1,0,0,1,1,0,1],
            [0,0,1,0,1,0,1,1],
            [0,0,0,1,0,1,1,1],
        ], dtype=np.uint8)

        return ExtremePointPolytope(E, B, use_D_sparse=use_D_sparse)
    elif dim == 2:
        E = np.array([
            [0,0],
            [1,0],
            [0,1],
            [1,1]
        ], float)

        # constraint ordering:
        # x=0,y=0,x=1,y=1
        B = np.array([
            [1,0,1,0],
            [1,1,0,0],
            [0,1,0,1],
            [0,0,1,1],
        ], dtype=np.uint8)

        return ExtremePointPolytope(E, B, use_D_sparse=use_D_sparse)
    
    if dim >= 4 and isinstance(dim, int):
        return d_cube_polytope(dim, use_D_sparse=use_D_sparse)
    
    else:
        raise NotImplementedError("Only dim=2, dim=3, and dim=4 supported in this helper")

@pytest.fixture
def dimensions():
    return [2,3,4]

@pytest.fixture
def use_D_sparse_options():
    return [False, True]

@pytest.fixture
def one_cut_corner_polys(dimensions, use_D_sparse_options):

    polys = []
    for use_D_sparse in use_D_sparse_options:
        for dimension in dimensions:
            poly_one_cut_corner = cube_polytope(dim=dimension, use_D_sparse=use_D_sparse)
            delta = 0.1
            d = poly_one_cut_corner.r
            ones = np.ones(d).tolist()
            poly_one_cut_corner.add_constraint(ones, d - delta)
            polys.append(poly_one_cut_corner)
    return polys

@pytest.fixture
def cube_polys(dimensions, use_D_sparse_options):
    return [cube_polytope(dim=dimension, use_D_sparse=use_D_sparse) for dimension in dimensions for use_D_sparse in use_D_sparse_options]

def test_cube_vertex_rank(cube_polys):
    '''Every vertex of the cube must lie on exactly r facets.'''
    for poly in cube_polys:
        counts = np.sum(poly.B, axis=0)
        assert np.all(counts == poly.r)
    
def test_adjacency_rule(cube_polys, one_cut_corner_polys):
    '''Adjacency rule: two vertices are adjacent iff they share exactly r-1 active constraints.'''
    for poly in cube_polys + one_cut_corner_polys:
        D = cast_to_dense_if_sparse(poly.D) 
        BTB = poly.B.T @ poly.B

        n = poly.E.shape[0]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                expected = (BTB[i, j] == poly.r - 1)
                assert D[i, j] == expected

def test_D(cube_polys, one_cut_corner_polys):
    '''Test that the D matrix is correct for both the original cube and after cutting in one corner.'''

    for poly in cube_polys:
        D = cast_to_dense_if_sparse(poly.D) 
        D_expected = np.array([[0,1,1,0],
                            [1,0,0,1],
                            [1,0,0,1],
                            [0,1,1,0]], dtype=np.uint8)
        if poly.r == 2:
            assert np.array_equal(D, D_expected), f"Expected D:\n{D_expected}\nGot:\n{D}"

    for poly in one_cut_corner_polys:
        D = cast_to_dense_if_sparse(poly.D) 
        D_expected = np.array([[0, 1, 1, 0, 0],
                            [1, 0, 0, 1, 0],
                            [1, 0, 0, 0, 1],
                            [0, 1, 0, 0, 1],
                            [0, 0, 1, 1, 0]], dtype=np.uint8)
        if poly.r == 2:
            assert np.array_equal(D, D_expected), f"Expected D:\n{D_expected}\nGot:\n{D}"

def test_cut_removes_corner(cube_polys):
    '''Cutting the cube with should remove the corner vertex.
    e.g. in dim=3, cutting with x+y+z <= 2.9 should remove the (1,1,1) vertex.
    '''
    for poly in cube_polys:
        delta = 0.1
        d = poly.r
        poly.add_constraint(np.ones(d).tolist(), d - delta)

        assert not np.any(np.all(np.isclose(poly.E, np.ones(d).tolist()), axis=1))

def test_new_vertices_exist(use_D_sparse_options):
    '''Cutting the cube with x+y+z <= 3-delta should create new vertices at (1,1,1-delta), (1,1-delta,1), and (1-delta,1,1).'''
    for use_D_sparse in use_D_sparse_options:
        poly = d_cube_polytope(dim=3, use_D_sparse=use_D_sparse)
        delta = 0.1
        d = poly.r
        poly.add_constraint(np.ones(d).tolist(), d - delta)

        expected = [
            [1,1,1-delta],
            [1,1-delta,1],
            [1-delta,1,1],
        ]

        for v in expected:
            assert np.any(np.all(np.isclose(poly.E, v, atol=1e-8), axis=1))

def test_rank_invariant_after_cut(one_cut_corner_polys):
    '''After cutting in one corner every vertex of the resulting polytope should still lie on exactly r facets.'''
    for poly in one_cut_corner_polys:
        counts = np.sum(poly.B, axis=0)
        assert np.all(counts == poly.r)

def test_adjacency_after_cut(one_cut_corner_polys):
    '''After cutting in one corner, the adjacency rule should still hold: two vertices are adjacent iff they share exactly r-1 active constraints.'''
    for poly in one_cut_corner_polys:
        D = cast_to_dense_if_sparse(poly.D)

        BTB = poly.B.T @ poly.B
        n = poly.E.shape[0]

        for i in range(n):
            for j in range(i+1, n):
                expected = (BTB[i,j] == poly.r - 1)
                assert D[i,j] == expected

def test_new_constraint_active(one_cut_corner_polys):
    '''After cutting in one corner, the new constraint should be active on all new vertices.'''

    for poly_one_cut_corner in one_cut_corner_polys:

        new_constraint_row = poly_one_cut_corner.B[-1]

        # at least 3 vertices must lie on new plane
        assert np.sum(new_constraint_row) >= poly_one_cut_corner.r

