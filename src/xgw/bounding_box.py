import itertools
import numpy as np
from scipy.optimize import linprog
import itertools
import ot

def is_in_convex_hull_lp(points, p, tol=1e-9):
    """
    Check if point p is in the convex hull of 'points' using LP.
    points: array (n, D)
    p: array (D,)
    returns: bool
    """
    points = np.asarray(points, dtype=float)
    p = np.asarray(p, dtype=float)
    n, D = points.shape
    if n == 0:
        return False
    # Objective is irrelevant (feasibility only)
    c = np.zeros(n)
    # constraints: points.T @ lambda = p, sum(lambda)=1
    A_eq = np.vstack([points.T, np.ones(n)])
    b_eq = np.concatenate([p, [1.0]])
    bounds = [(0.0, None)] * n
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return False
    # numerical check
    recon = res.x @ points
    return np.linalg.norm(recon - p, ord=np.inf) <= max(tol, 1e-12)

def add_if_outside_lp(points, p, tol=1e-9):
    if is_in_convex_hull_lp(points, p, tol=tol):
        return points, False
    else:
        new_points = np.vstack([points, p.reshape(1, -1)])
        return new_points, True

def enumerate_vertices(A, b):
    """
    Enumerate all vertices of a polytope {x | A x <= b} by intersecting d hyperplanes.
    Returns an array of vertices (n_vertices, d).
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    m, d = A.shape
    vertices = []

    for idx in itertools.combinations(range(m), d):
        A_sub = A[list(idx)]
        b_sub = b[list(idx)]
        if np.linalg.matrix_rank(A_sub) < d:
            continue
        try:
            x = np.linalg.solve(A_sub, b_sub)
        except np.linalg.LinAlgError:
            continue
        if np.all(A @ x <= b + 1e-12):
            vertices.append(x)

    if vertices:
        return np.array(vertices)
    else:
        return np.empty((0, A.shape[1]))


def add_halfspace_and_max_norm(A, b, e, h):
    """
    Add a new halfspace e^T x <= h to existing halfspaces A x <= b,
    enumerate all vertices, and return max squared L2 norm and vertex.
    """
    e = np.asarray(e, dtype=float).reshape(1, -1)
    h = float(h)

    if A.size == 0:
        A_new = e
        b_new = np.array([h])
    else:
        A_new = np.vstack([A, e])
        b_new = np.append(b, h)

    vertices = enumerate_vertices(A_new, b_new)
    if vertices.size == 0:
        return None, None, A_new, b_new  # polytope empty

    norms2 = np.sum(vertices**2, axis=1)
    max_idx = np.argmax(norms2)
    return norms2[max_idx], vertices[max_idx], A_new, b_new

def construct_basis_eij_in_PI(space_x, space_y):
    N, dx = space_x.shape
    M, dy = space_y.shape
    e_ijs_in_PI = np.zeros((N, M, dx, dy))
    assert dx == dy
    for i in range(dx):
        for j in range(dy):
            e_ijs_in_PI[:,:, i, j] = np.outer(space_x[:,i], space_y[:,j])
    e_ijs_in_PI /= np.linalg.norm(e_ijs_in_PI[:,:,0,0])
    return e_ijs_in_PI.reshape(N, M, dx * dy)


def construct_cost_at_point_in_P_PI(point_in_p_PI, e_ijs_in_PI):
    M, N, d2 = e_ijs_in_PI.shape
    d2_ = len(point_in_p_PI)
    assert d2 == d2_, f"dimension mismatch between {d2} and {d2_}"
    return (point_in_p_PI.reshape(1,1,-1) * e_ijs_in_PI).sum(-1)


def compute_e_h_from_cost_e(marginal_a, marginal_b, cost_e):
    cost, log = ot.emd2(marginal_a, marginal_b, M=-cost_e, log=True)
    return cost, log['T']


def project_pi_on_p_pi(space_x, space_y, pi):
    x = np.zeros((len(space_x.T), len(space_y.T)))
    for i in range(len(space_x.T)):
        for j in range(len(space_y.T)):
            x[i,j] = np.einsum('k,l,kl->', space_x[:,i], space_y[:,j], pi)
    return x.flatten()


def update_bdb(e, p_pi_neg, p_pi_plus, c_pi_neg, c_pi_plus, pi_opt, marginal_a, marginal_b, space_x, space_y):
    e_ijs_in_PI = construct_basis_eij_in_PI(space_x, space_y)
    cost_e = construct_cost_at_point_in_P_PI(e, e_ijs_in_PI)
    pi_new, h_new = compute_e_h_from_cost_e(marginal_a, marginal_b, cost_e)
    x = project_pi_on_p_pi(space_x, space_y, pi_new)
    p_pi_neg, added = add_if_outside_lp(p_pi_neg, x) # TODO: handle condition when added is False
    c_pi_neg_test = np.linalg.norm(x)**2
    if c_pi_neg < c_pi_neg_test:
        pi_opt = pi_new
        c_pi_neg = c_pi_neg_test
    A, b = p_pi_plus
    max_norm2, _, A, b = add_halfspace_and_max_norm(A, b, e, h_new)
    p_pi_plus = (A, b)
    c_pi_plus = min(c_pi_plus, max_norm2)
    return p_pi_neg, p_pi_plus, c_pi_neg, c_pi_plus, pi_opt

    