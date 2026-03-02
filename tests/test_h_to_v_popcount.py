import numpy as np
import pytest
from xgw.h_to_v_popcount import ExtremePointPolytope3D


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def cube_polytope():
    """
    Create unit cube with correct B matrix.
    """

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

    return ExtremePointPolytope3D(E, B)

@pytest.fixture
def poly_one_cut_corner():
    poly_one_cut_corner = cube_polytope()
    poly_one_cut_corner.add_constraint([1,1,1], 2.9)
    return poly_one_cut_corner

@pytest.fixture
def poly_cube():
    return cube_polytope()

# ------------------------------------------------------------
# TEST 1 — cube invariants
# ------------------------------------------------------------
def test_cube_vertex_rank(poly_cube):
    # every vertex must lie on exactly r=3 facets
    counts = np.sum(poly_cube.B, axis=0)
    assert np.all(counts == 3)


# ------------------------------------------------------------
# TEST 2 — adjacency rule (B^T B)
# ------------------------------------------------------------
def test_adjacency_rule(poly_one_cut_corner, poly_cube):
    for poly in [poly_one_cut_corner, poly_cube]:

        BTB = poly.B.T @ poly.B

        n = poly.E.shape[0]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                expected = (BTB[i, j] == poly.r - 1)
                assert poly.D[i, j] == expected


# ------------------------------------------------------------
# TEST 3 — cutting removes (1,1,1)
# ------------------------------------------------------------
def test_cut_removes_corner():
    poly = cube_polytope()
    poly.add_constraint([1,1,1], 2.9)

    assert not np.any(np.all(np.isclose(poly.E, [1,1,1]), axis=1))


# ------------------------------------------------------------
# TEST 4 — new vertices created correctly
# ------------------------------------------------------------
def test_new_vertices_exist():
    poly = cube_polytope()
    poly.add_constraint([1,1,1], 2.9)

    expected = [
        [1,1,0.9],
        [1,0.9,1],
        [0.9,1,1],
    ]

    for v in expected:
        assert np.any(np.all(np.isclose(poly.E, v, atol=1e-8), axis=1))


# ------------------------------------------------------------
# TEST 5 — rank invariant AFTER CUT
# ------------------------------------------------------------
def test_rank_invariant_after_cut(poly_one_cut_corner):
    counts = np.sum(poly_one_cut_corner.B, axis=0)
    assert np.all(counts == 3)


# ------------------------------------------------------------
# TEST 6 — adjacency still consistent
# ------------------------------------------------------------
def test_adjacency_after_cut(poly_one_cut_corner):
    poly = poly_one_cut_corner

    BTB = poly_one_cut_corner.B.T @ poly_one_cut_corner.B
    n = poly_one_cut_corner.E.shape[0]

    for i in range(n):
        for j in range(i+1, n):
            expected = (BTB[i,j] == poly.r - 1)
            assert poly_one_cut_corner.D[i,j] == expected


# ------------------------------------------------------------
# TEST 7 — new constraint active on new vertices
# ------------------------------------------------------------
def test_new_constraint_active(poly_one_cut_corner):

    new_constraint_row = poly_one_cut_corner.B[-1]

    # at least 3 vertices must lie on new plane
    assert np.sum(new_constraint_row) >= 3