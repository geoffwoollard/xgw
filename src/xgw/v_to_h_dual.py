import numpy as np
import itertools

from xgw.utils import canonicalize_vertices
from xgw.hyperplane_approx import DoubleDescription, build_B_from_H_and_V


def center_polytope(vertices):
    """
    Shift polytope so origin is inside (centroid shift).
    """
    center = np.mean(vertices, axis=0)
    vertices_shifted = vertices - center
    return vertices_shifted, center

def center_polytope_facets(A, b, center):
    """
    Shift facets to match vertex shift.
    """
    # new_facets = []
    # for f in facets:
    #     a = f["a"]
    #     b = f["b"]
    #     b_new = b - a @ center
    #     new_facets.append({
    #         "a": a.copy(),
    #         "b": b_new
    #     })
    # return new_facets
    return b - A @ center

def normalize_facets(A, b, tol=1e-12):
    """
    Convert all facets to a^T x <= 1 form.
    """
    # new_facets = []

    # for f in facets:
    #     a = f["a"]
    #     b = f["b"]

    #     if abs(b) < tol:
    #         raise ValueError("Facet too close to origin; cannot normalize")

    #     a_new = a / b
    #     b_new = 1.0

    #     new_facets.append({
    #         "a": a_new,
    #         "b": b_new
    #     })

    # return new_facets
    if any(np.abs(b) < tol):
        raise ValueError("Facet too close to origin; cannot normalize")
    
    A_normalized = A / b[:, np.newaxis]
    b_normalized = np.ones_like(b)
    return A_normalized, b_normalized

def primal_to_dual(A):
    """
    facets: list of {"a", "b"} with b=1

    returns:
        dual_vertices: (F, d)
    """
    # dual_vertices = np.array([f["a"] for f in facets])
    # return dual_vertices
    dual_vertices = A
    return dual_vertices

def dual_to_primal_facets(dual_vertices):
    """
    dual_vertices: (N, d)

    returns facets in form a^T x <= 1
    """
    # facets = []
    # for y in dual_vertices:
    #     facets.append({
    #         "a": y.copy(),
    #         "b": 1.0
    #     })
    # return facets
    A = dual_vertices
    b = np.ones(len(dual_vertices))
    return A, b

def recover_vertices_from_facets(A, b, d, tol=1e-10):
    """
    Compute vertices by intersecting combinations of d facets.
    (brute force, OK for testing)
    """

    vertices = []
    facets = [{"a": A[i], "b": b[i]} for i in range(len(b))]

    for combo in itertools.combinations(facets, d):
        A = np.stack([f["a"] for f in combo])
        b = np.ones(d)

        if abs(np.linalg.det(A)) < tol:
            continue

        x = np.linalg.solve(A, b)

        # check feasibility
        if all(f["a"] @ x <= f["b"] + tol for f in facets):
            vertices.append(x)

    if not vertices:
        return np.zeros((0, d))

    V = np.unique(np.round(vertices, 10), axis=0)
    return V


def setup_dual_polytope(vertices, A, b, implementation):
    d = vertices.shape[1]

    # Step 1: center
    vertices, center = center_polytope(vertices)
    b = center_polytope_facets(A, b, center)

    # Step 2: normalize facets
    A, b = normalize_facets(A, b)

    # Step 3: primal → dual
    dual_vertices = primal_to_dual(A)

    # Step 4: add halfspace in dual
    # x_new becomes inequality: x_new^T y <= 1
    # initialize h_to_v data structure from vertices

    if implementation == 'cdd':
        dd_dual_polytope = DoubleDescription(implementation=implementation)
        dd_dual_polytope.add_V(dual_vertices.tolist())
    elif implementation == 'h_to_v_popcount':
        dual_A = vertices
        dual_b = np.ones(len(vertices))
        dual_B_initialization = build_B_from_H_and_V(dual_A, dual_b, dual_vertices)
        dd_dual_polytope = DoubleDescription(implementation='h_to_v_popcount', V_initialization=dual_vertices, B_initialization=dual_B_initialization)
        dd_dual_polytope.V = dual_vertices.tolist()
        dd_dual_polytope.H = [dual_A, dual_b]
    else:
        raise ValueError(f"Unknown implementation: {implementation}")

    return dd_dual_polytope, center, d

def add_halfspace_dual(x_new, dd_dual_polytope):
    """
    dual_vertices: (N, d)
    x_new: (d,)

    returns new_dual_vertices: (N', d)
    """
    a_new = x_new
    b_new = np.array([1.0])
    dd_dual_polytope.add_H([[a_new, b_new]])
    return dd_dual_polytope

def add_vertex_via_dual(x_new, center, dd_dual_polytope, d):
    """
        signature: dual_vertices -> new_dual_vertices
    """

    # Step 1-3: setup dual polytope
    dd_dual_polytope = add_halfspace_dual(x_new - center, dd_dual_polytope)
    new_dual_vertices = np.array(dd_dual_polytope.V)
    # Step 5: dual → primal facets
    A, b = dual_to_primal_facets(new_dual_vertices)
    # Step 6: recover vertices.
    A_dual, b_dual = dd_dual_polytope.H
    vertices_from_dual = A_dual / b_dual[:, np.newaxis]

    def _test_vertex_recovery():
        vertices_from_dual = canonicalize_vertices(vertices_from_dual) #TODO: beware of this. just use for tests but do not truncate vertices in main code, as it can cause issues with precision and uniqueness.
        optional_recovered_vertices = recover_vertices_from_facets(A, b, d)
        optional_recovered_vertices = canonicalize_vertices(optional_recovered_vertices)
        assert np.allclose(optional_recovered_vertices, vertices_from_dual), f"Dual vertices should be the a vectors of the dual facets, but they differ: {A_dual} vs {vertices_from_dual}"
        assert np.allclose(b_dual, 1.0), f"Dual facets should be in form a^T x <= 1, but b_dual is not all 1s: {b_dual}"

    vertices = vertices_from_dual

    # Step 7: shift back
    vertices = vertices + center
    b = center_polytope_facets(A, b, -center)
    return vertices, A, b, dd_dual_polytope




