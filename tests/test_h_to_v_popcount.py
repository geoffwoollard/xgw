import numpy as np
import pytest
from xgw.h_to_v_popcount import ExtremePointPolytope3D


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def d_cube_polytope(dim):
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

    return ExtremePointPolytope3D(E, B, dim=dim)

def cube_polytope(dim):
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

        return ExtremePointPolytope3D(E, B, dim=dim)
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
            [0,0,1,0],
            [0,1,0,1],
        ], dtype=np.uint8)

        return ExtremePointPolytope3D(E, B, dim=dim)
    
    elif dim >= 4 and isinstance(dim, int):
        return d_cube_polytope(dim)
    
    else:
        raise NotImplementedError("Only dim=2, dim=3, and dim=4 supported in this helper")

@pytest.fixture
def dimensions():
    return [3, 4, 5, 6, 7, 8, 9, 10]

@pytest.fixture
def one_cut_corner_polys(dimensions):

    polys = []
    for dimension in dimensions:
        poly_one_cut_corner = cube_polytope(dim=dimension)
        delta = 0.1
        d = poly_one_cut_corner.r
        poly_one_cut_corner.add_constraint(np.ones(d).tolist(), d - delta)
        polys.append(poly_one_cut_corner)
    return polys

@pytest.fixture
def cube_polys(dimensions):
    return [cube_polytope(dim=dimension) for dimension in dimensions]

# ------------------------------------------------------------
# TEST 1 — cube invariants
# ------------------------------------------------------------
def test_cube_vertex_rank(cube_polys):
    for poly in cube_polys:
        counts = np.sum(poly.B, axis=0)
        assert np.all(counts == poly.r)
    # every vertex must lie on exactly r=3 facets



# ------------------------------------------------------------
# TEST 2 — adjacency rule (B^T B)
# ------------------------------------------------------------
def test_adjacency_rule(one_cut_corner_polys, cube_polys):
    for poly in one_cut_corner_polys + cube_polys:

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
def test_cut_removes_corner(cube_polys):
    for poly in cube_polys:
        delta = 0.1
        d = poly.r
        poly.add_constraint(np.ones(d).tolist(), d - delta)

        assert not np.any(np.all(np.isclose(poly.E, np.ones(d).tolist()), axis=1))


# ------------------------------------------------------------
# TEST 4 — new vertices created correctly
# ------------------------------------------------------------
def test_new_vertices_exist():
    poly = d_cube_polytope(dim=3)
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


# ------------------------------------------------------------
# TEST 5 — rank invariant AFTER CUT
# ------------------------------------------------------------
def test_rank_invariant_after_cut(one_cut_corner_polys):
    for poly in one_cut_corner_polys:
        counts = np.sum(poly.B, axis=0)
        assert np.all(counts == poly.r)


# ------------------------------------------------------------
# TEST 6 — adjacency still consistent
# ------------------------------------------------------------
def test_adjacency_after_cut(one_cut_corner_polys):
    for poly in one_cut_corner_polys:

        BTB = poly.B.T @ poly.B
        n = poly.E.shape[0]

        for i in range(n):
            for j in range(i+1, n):
                expected = (BTB[i,j] == poly.r - 1)
                assert poly.D[i,j] == expected


# ------------------------------------------------------------
# TEST 7 — new constraint active on new vertices
# ------------------------------------------------------------
def test_new_constraint_active(one_cut_corner_polys):

    for poly_one_cut_corner in one_cut_corner_polys:

        new_constraint_row = poly_one_cut_corner.B[-1]

        # at least 3 vertices must lie on new plane
        assert np.sum(new_constraint_row) >= poly_one_cut_corner.r