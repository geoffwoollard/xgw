import numpy as np
import pytest
from itertools import product

from xgw.h_to_v_popcount import ExtremePointPolytopeSparse, masks_from_B, masks_from_B_vectorized


def cube_mask(v):
    # bits 0..d-1: coordinate == 1 ; bits d..2d-1: coordinate == 0
    d = len(v)
    m = 0
    for i in range(d):
        if v[i] == 1:
            m |= 1 << i
        if v[i] == 0:
            m |= 1 << (d + i)
    return m

def unit_cube(dim):
    # Vertices
    E = np.array(list(product([0,1], repeat=dim)), dtype=float)

    # Halfspace form
    A = np.vstack([np.eye(dim), -np.eye(dim)])
    b = np.concatenate([np.ones(dim), np.zeros(dim)])

    return E, A, b

def cube_polytope(dim):
    """
    Create unit cube with correct B matrix.
    """

    E, A, b = unit_cube(dim)

    masks = [cube_mask(v) for v in E]
    poly = ExtremePointPolytopeSparse(E, masks, A, b)
    return poly

def test_masks_from_B(dimensions):
    '''Test that masks_from_B and masks_from_B_vectorized give the same results, and that they match the expected cube masks for the unit cube.'''

    for dim in dimensions:
        E, A, b = unit_cube(dim)

        # compute boolean incidence matrix B: shape (m_constraints, n_vertices)
        # lhs shape (n_vertices, m_constraints) so transpose to (m, n_vertices)
        lhs = E @ A.T  # (n_vertices, m_constraints)
        B = np.isclose(lhs, b)  # broadcasting: (n_vertices, m_constraints)
        B = B.T  # (m_constraints, n_vertices)

        masks = masks_from_B(B)
        masks_vec = masks_from_B_vectorized(B)
        assert np.array_equal(masks, masks_vec), f"masks_from_B and masks_from_B_vectorized give different results for dimension {dim}"
        masks_cube = [cube_mask(v) for v in E]
        assert np.array_equal(masks, masks_cube), f"Expected masks {masks_cube} but got {masks} for dimension {dim}"
        assert np.array_equal(masks_vec, masks_cube), f"Expected masks {masks_cube} but got {masks_vec} for dimension {dim}"

@pytest.fixture
def dimensions():
    return [2,3,4,5,6,7,8,9,10]

@pytest.fixture
def delta():
    return 0.1

@pytest.fixture
def one_cut_corner_polys(dimensions, delta):
    polys = []
    for dimension in dimensions:
        poly_one_cut_corner = cube_polytope(dim=dimension)
        d = poly_one_cut_corner.r
        ones = np.ones(d).tolist()
        poly_one_cut_corner.add_constraint(ones, d - delta)
        polys.append(poly_one_cut_corner)
    return polys

@pytest.fixture
def double_one_cut_corner_polys(dimensions, delta):
    polys = []
    for dimension in dimensions:
        poly_one_cut_corner = cube_polytope(dim=dimension)
        d = poly_one_cut_corner.r
        ones = np.ones(d).tolist()
        poly_one_cut_corner.add_constraint(ones, d - delta)
        poly_one_cut_corner.add_constraint(ones, d - 2*delta)
        polys.append(poly_one_cut_corner)
    return polys

@pytest.fixture
def all_corners_except_one_polys(dimensions, delta):
    polys = []
    for dimension in dimensions:
        poly_one_cut_corner = cube_polytope(dim=dimension)
        d = poly_one_cut_corner.r
        ones = np.ones(d).tolist()
        poly_one_cut_corner.add_constraint(ones, delta)
        polys.append(poly_one_cut_corner)
    return polys

@pytest.fixture
def double_all_corners_except_one_polys(dimensions, delta):
    polys = []
    for dimension in dimensions:
        poly_one_cut_corner = cube_polytope(dim=dimension)
        d = poly_one_cut_corner.r
        ones = np.ones(d).tolist()
        poly_one_cut_corner.add_constraint(ones, delta)
        poly_one_cut_corner.add_constraint(ones, delta/2)
        polys.append(poly_one_cut_corner)
    return polys

@pytest.fixture
def cube_polys(dimensions):
    return [cube_polytope(dim=dimension) for dimension in dimensions]

def test_cube_vertex_rank(cube_polys):
    """Every vertex of the cube must lie on exactly r facets.
    When using bitmask representation, each mask's bit_count() gives
    the number of active constraints for that vertex.
    """
    for poly in cube_polys:
        counts = np.array([m.bit_count() for m in poly.masks], dtype=int)
        assert np.all(counts == poly.r)

def test_assert_symmetric_D(cube_polys, one_cut_corner_polys, double_one_cut_corner_polys, all_corners_except_one_polys, double_all_corners_except_one_polys):
    for poly in cube_polys + one_cut_corner_polys + double_one_cut_corner_polys + all_corners_except_one_polys + double_all_corners_except_one_polys:
        M = poly.D
        assert np.all(M == M.T), "Adjacency matrix not symmetric"

def test_vertex_count(cube_polys, one_cut_corner_polys, double_one_cut_corner_polys, all_corners_except_one_polys, double_all_corners_except_one_polys):
    for poly in cube_polys:
        assert len(poly.E) == 2**poly.r, f"Expected {2**poly.r} vertices for the cube, but got {len(poly.E)} for E={poly.E}"

    for poly in one_cut_corner_polys + double_one_cut_corner_polys:
        assert len(poly.E) == 2**poly.r - 1 + poly.r, f"Expected {2**poly.r - 1 + poly.r} vertices after cutting one corner of the cube, but got {len(poly.E)} for E={poly.E}"

    for poly in all_corners_except_one_polys + double_all_corners_except_one_polys:
        assert len(poly.E) == 1 + poly.r, f"Expected {1 + poly.r} vertices after cutting off all but one corner of the cube, but got {len(poly.E)} for E={poly.E}"

def test_adjacency_rule(cube_polys, one_cut_corner_polys, double_one_cut_corner_polys, all_corners_except_one_polys, double_all_corners_except_one_polys):
    """Adjacency rule using bitmask representation:
    two vertices are adjacent iff they share exactly r-1 active constraints.
    """
    for poly in one_cut_corner_polys + double_one_cut_corner_polys + cube_polys + all_corners_except_one_polys + double_all_corners_except_one_polys:
        n = len(poly.masks)
        for i in range(n):
            for j in range(i + 1, n):
                # number of shared active constraints = popcount(mask_i & mask_j)
                shared = (poly.masks[i] & poly.masks[j]).bit_count()
                expected = (shared == poly.r - 1)
                assert bool(poly.D[i, j]) == expected, (
                    f"Adjacency mismatch for poly.r={poly.r} vertices {i},{j}: "
                    f"shared={shared}, expected={expected}"
                )

def test_active_constraints(cube_polys, one_cut_corner_polys, double_one_cut_corner_polys, all_corners_except_one_polys, double_all_corners_except_one_polys):
    for poly in one_cut_corner_polys + double_one_cut_corner_polys + cube_polys + all_corners_except_one_polys + double_all_corners_except_one_polys:
        for m in poly.masks:
            assert m.bit_count() == poly.r, f"Mask {m} does not have exactly r active constraints"

def test_cut_removes_corner(one_cut_corner_polys, double_one_cut_corner_polys, all_corners_except_one_polys, double_all_corners_except_one_polys):
    '''Cutting the cube with should remove the corner vertex.
    e.g. in dim=3, cutting with x+y+z <= 2.9 should remove the all ones vertex: (1, 1, ..., 1).
    '''
    for poly in one_cut_corner_polys + double_one_cut_corner_polys + all_corners_except_one_polys + double_all_corners_except_one_polys:
        assert not np.any(np.all(np.isclose(poly.E, np.ones(poly.r).tolist()), axis=1))

def test_new_vertices_on_plane(one_cut_corner_polys, double_one_cut_corner_polys, all_corners_except_one_polys, double_all_corners_except_one_polys, delta):
    '''Cutting the cube with x+y+z <= 3-delta should create new vertices at (1,1,1-delta), (1,1-delta,1), and (1-delta,1,1).
    Cutting the cube with x+y+z <= delta should create new vertices at (delta,0,0), (0,delta,0), and (0,0,delta).
    '''
    
    for poly_list, _delta in zip([one_cut_corner_polys, double_one_cut_corner_polys], [delta, 2*delta]):
        for poly in poly_list:
            vals = poly.E @ np.ones(poly.r)
            on_plane = np.isclose(vals, poly.r - _delta, atol=1e-8)
            assert np.sum(on_plane) == poly.r, f"Expected {poly.r} new vertices to lie on the new plane, but got {np.sum(on_plane)}"

    for poly_list, _delta in zip([all_corners_except_one_polys, double_all_corners_except_one_polys], [delta, delta/2]):
        for poly in poly_list:
            vals = poly.E @ np.ones(poly.r)
            on_plane = np.isclose(vals, _delta, atol=1e-8)
            assert np.sum(on_plane) == poly.r, f"Expected {poly.r} new vertices to lie on the new plane, but got {np.sum(on_plane)}"

def test_new_vertices_exist(one_cut_corner_polys, double_one_cut_corner_polys, all_corners_except_one_polys, double_all_corners_except_one_polys, delta):
    """
    Cutting the d-cube with sum(x_i) <= d - delta
    should create d new vertices at

        1 - delta in exactly one coordinate,
        1 elsewhere.
    """
    for poly_list, _delta in zip([one_cut_corner_polys, double_one_cut_corner_polys], [delta, 2*delta]):
        for poly in poly_list:
            d = poly.r

            # Expected new vertices: 1 - delta in one coordinate
            expected = []
            for i in range(d):
                v = np.ones(d)
                v[i] = 1 - _delta
                expected.append(v)

            # Check each expected vertex exists
            for v in expected:
                assert np.any(
                    np.all(np.isclose(poly.E, v, atol=1e-8), axis=1)
                ), f"Expected new vertex {v} not found in E:\n{poly.E}"

    for poly_list, _delta in zip([all_corners_except_one_polys, double_all_corners_except_one_polys], [delta, delta/2]):
        for poly in poly_list:
            d = poly.r

            # Expected new vertices: 1 - delta in one coordinate
            expected = []
            for i in range(d):
                v = np.zeros(d)
                v[i] = _delta
                expected.append(v)

            # Check each expected vertex exists
            for v in expected:
                assert np.any(
                    np.all(np.isclose(poly.E, v, atol=1e-8), axis=1)
                ), f"Expected new vertex {v} not found in E:\n{poly.E}"

def test_rank_invariant_after_cut(one_cut_corner_polys, double_one_cut_corner_polys, all_corners_except_one_polys, double_all_corners_except_one_polys):
    """After cutting in one corner every vertex of the resulting polytope
    should still lie on exactly r facets (check via popcount of masks)."""
    for poly in one_cut_corner_polys + double_one_cut_corner_polys + all_corners_except_one_polys:
        counts = np.array([m.bit_count() for m in poly.masks], dtype=int)
        assert np.all(counts == poly.r), f"Some vertices do not lie on {poly.r} facets"

def test_adjacency_after_cut(one_cut_corner_polys, double_one_cut_corner_polys, all_corners_except_one_polys, double_all_corners_except_one_polys):
    """After cutting in one corner, adjacency rule should still hold:
    two vertices are adjacent iff they share exactly r-1 active constraints."""
    for poly in one_cut_corner_polys + double_one_cut_corner_polys + all_corners_except_one_polys:
        n = len(poly.masks)
        for i in range(n):
            for j in range(i + 1, n):
                shared = (poly.masks[i] & poly.masks[j]).bit_count()
                expected = (shared == poly.r - 1)
                assert bool(poly.D[i, j]) == expected, (
                    f"Adjacency mismatch after cut for poly.r={poly.r} vertices {i},{j}: "
                    f"shared={shared}, expected={expected}"
                )

def test_new_constraint_active(one_cut_corner_polys, double_one_cut_corner_polys, all_corners_except_one_polys, double_all_corners_except_one_polys, delta):
    '''After cutting in one corner, the new constraint should be active on all new vertices.'''

    for poly_one_cut_corner in one_cut_corner_polys + double_one_cut_corner_polys + all_corners_except_one_polys + double_all_corners_except_one_polys:

        sum_new_constraint_row = poly_one_cut_corner.masks[-1].bit_count()

        # at least r vertices must lie on new plane
        assert sum_new_constraint_row >= poly_one_cut_corner.r

def test_new_vertices_triangle(one_cut_corner_polys, double_one_cut_corner_polys, all_corners_except_one_polys, double_all_corners_except_one_polys, delta):
    '''Cutting the cube with x+y+z <= 3-delta or x+y+z <= delta should create new vertices that are mutually adjacent (form a triangle in dim=3, tetrahedron in dim=4, etc).'''
    for poly_list, _delta in zip([one_cut_corner_polys, double_one_cut_corner_polys], [delta, 2*delta]):
        for poly in poly_list:
            d = poly.r

            vals = poly.E @ np.ones(d)
            new_idx = np.where(np.isclose(vals, d - _delta))[0]

            subgraph = poly.D[np.ix_(new_idx, new_idx)]

            # triangle adjacency
            expected = np.ones((d,d), dtype=int) - np.eye(d, dtype=int)
            assert np.all(subgraph == expected)

    for poly_list, _delta in zip([all_corners_except_one_polys, double_all_corners_except_one_polys], [delta, delta/2]):
        for poly in poly_list:
            d = poly.r

            vals = poly.E @ np.ones(d)
            new_idx = np.where(np.isclose(vals, _delta))[0]

            subgraph = poly.D[np.ix_(new_idx, new_idx)]

            # triangle adjacency
            expected = np.ones((d,d), dtype=int) - np.eye(d, dtype=int)
            assert np.all(subgraph == expected)

if __name__ == "__main__":
    test_masks_from_B([2, 3, 4, 5,6,7,8,9,10])