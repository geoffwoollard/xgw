import numpy as np
from xgw.Hyperplane_approx import DoubleRepresentation
from numba import njit

def segment_plane_intersection(v0, v1, a, b, tol=1e-12):
    """
    Intersection between segment [v0, v1] and hyperplane a^T x = b.
    
    Returns:
        - intersection point (np.ndarray) if it exists within the segment
        - None if no intersection
        - "in_plane" if the whole segment lies in the plane
    """
    v0 = np.asarray(v0, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    a  = np.asarray(a,  dtype=float)

    d = v1 - v0
    denom = np.dot(a, d)
    num = b - np.dot(a, v0)

    if abs(denom) < tol:
        if abs(num) < tol:
            return "in_plane"
        else:
            return None

    t = num / denom

    if -tol <= t <= 1 + tol:
        return v0 + t * d
    else:
        return None


def affine_subspace_basis(points, normal_vec):
    """
    points: array of shape (n_points, d)
    normal_vec: vector normal to the plane supporting the points
    Returns:
        Q : orthonormal basis (d, k) for the subspace
    """
    dim = len(normal_vec)
    points = np.asarray(points, float)
    
    Q,R = np.linalg.qr(normal_vec.reshape((dim,1)), mode = 'complete')
    print (R[0,0])
    assert np.allclose(np.abs(R[0,0]), 1)
    Q = Q[:,1:]

    return  Q


def project_to_subspace(x,  Q):
    """
    Project x into coordinates of the subspace.
    Returns coordinates in R^(d-1)
    """
    return Q.T @ x 


def reindex_pairs(pairs, excluded, n):
    excluded = np.asarray(excluded)

    # Step 1: mask of kept indices
    keep = np.ones(n, dtype=bool)
    keep[excluded] = False

    # Step 2: mapping old index → new index
    # cumulative count of kept elements
    new_index = np.full(n, -1, dtype=int)
    new_index[keep] = np.arange(keep.sum())

    # Step 3: apply mapping to pairs
    return new_index[pairs]

def reindex_pairs_with_new(pairs, n_old, excluded, n_new):
    excluded = np.asarray(excluded)

    # --- Old vertices mapping ---
    keep = np.ones(n_old, dtype=bool)
    keep[excluded] = False

    n_kept = keep.sum()

    old_mapping = np.full(n_old, -1, dtype=int)
    old_mapping[keep] = np.arange(n_kept)

    # --- New vertices mapping ---
    # New indices are n_old ... n_old + n_new - 1
    new_mapping = np.arange(n_kept, n_kept + n_new)

    # --- Full mapping table ---
    mapping = np.concatenate([old_mapping, new_mapping])

    # --- Apply to all pairs ---
    return mapping[pairs]

@njit
def compute_constraints_matching(vertex_test, dim):
    # vertex_test of dimension n_constraints by n_points, boolean array of points solving constraints
    s = vertex_test.shape[1]
    finall_mat = np.zeros((s, s), dtype=np.bool_)
    for i in range (s-1):
        for j in range(i,s):
            common_constr = np.logical_and(vertex_test[:,i], vertex_test[:,j])
            # detecting the points solving the same dim-1 constraints
            finall_mat[i,j] =  np.sum(common_constr) == dim-1
    return finall_mat


def find_edges(points):
    dim = points[0].shape[0]
    if dim==1:
        E_new = np.array([0, 1])
    else:
        dd_sub = DoubleRepresentation()
        dd_sub.add_V(points)
        A_sub, b_sub = dd_sub.H
        dd_sub.H_to_V()
        # normalizing normal vectors
        c = 1/np.linalg.norm(A_sub, axis=-1)
        b_sub *= c
        A_sub *= np.tile(c,(A_sub.shape[1], 1)).T
        print( f' number of points not changed: {len(points)- len(dd_sub.V)==0}')
        # linking vertices and constraints
        vertex_test = np.isclose(A_sub @ np.array(points).T - b_sub[:, np.newaxis], 0)
        # checking that each vertex solves at least dim constraints
        print(vertex_test.sum(0))
        # print((A_sub @ np.array(points).T - b_sub[:, np.newaxis]).T)
        assert (vertex_test.sum(0) >= dim).all()
        # two vertices on a same edge solve the same dim-1 constraints 
        finall_mat = compute_constraints_matching(vertex_test, dim)
        E_new = np.transpose(np.nonzero(finall_mat))
    return E_new

def update_edges_with_new_halfplane(V, E, A, b, a_new, b_new):
    # check if  a_new is normalized (can remove later)
    a_norm = np.linalg.norm(a_new)
    a_new /= a_norm
    b_new /= a_norm
    # find excluded vertices
    v_excluded_bool = a_new.dot(V.T) < b_new
    v_excluded_idx = np.arange(V.shape[0])[v_excluded_bool]
    # print("excluded vertices:", v_excluded_idx)

    # find edges for new vertices
        # for each edge find v_new vertices
            # ensure new vertex inside (satisfies all old halfplanes)
        # and update old edges

    V_new = {}
    v_new_idx = len(V)
    E_old_updated = []
    for v_idx in v_excluded_idx:
        r, c = np.where(E == v_idx)
        for edge_idx in range(len(r)):
            idx_of_v_excluded = E[r[edge_idx], c[edge_idx]]
            idx_other = E[r[edge_idx], 1-c[edge_idx]]
            if (idx_of_v_excluded in v_excluded_idx and idx_other in v_excluded_idx):
                pass
            else:
                v_new = segment_plane_intersection(V[idx_of_v_excluded], V[idx_other], a_new, b_new)
                # check v inside all old halfplanes
                tol = 1e-12
                v_inside = A @ v_new <= b + tol
                if v_inside.all():
                    V_new[v_new_idx] = v_new
                    E_old_updated.append([v_new_idx, idx_other])
                    v_new_idx += 1
    E_old_updated = np.array(E_old_updated)
    # print("E_old_updated:", E_old_updated)
    # print("V_new:", V_new)
    # create new edges between new vertices
        # transform new vertices to full rank subspace
    points = np.stack(list(V_new.values()))
    # print("points", points)
    Q = affine_subspace_basis(points, a_new)
    # print("p0:\n", p0)
    # print("Q (basis for plane):\n", Q)
    # print("Q^T Q:\n", Q.T @ Q)

    new_points = []
    for p in points:
        coords = project_to_subspace(p, Q)
        new_points.append(coords)
        # print(p, "→", coords)

    # find edges between the new points
    E_new = find_edges(new_points)
    # re-index V and E accordingly
    delta_v_idx = len(V) - len(v_excluded_idx)
    E_new = E_new + delta_v_idx
    # print("re-indexed E_new:\n", E_new)
    # E_old_updated = E_old_updated + delta_v_idx
    # print("re-indexed E_old_updated:\n", E_old_updated)


        # remove old edges with excluded vertices
    E_reindexed = reindex_pairs(E, v_excluded_idx, len(V))
    valid = (E_reindexed >= 0).all(axis=1)
    E_reindexed_clean = E_reindexed[valid]
    # print("E_reindexed_clean:\n", E_reindexed_clean)

    E_old_updated_reindexed = reindex_pairs_with_new(E_old_updated, len(V), v_excluded_idx, len(V_new))
    valid = (E_old_updated_reindexed >= 0).all(axis=1)
    E_old_updated_reindexed_clean = E_old_updated_reindexed[valid]
    # print("E_old_updated_reindexed_clean:\n", E_old_updated_reindexed_clean)

        # combine all edges
    E_final = np.vstack([E_reindexed_clean, E_new, E_old_updated_reindexed_clean])
    # print("E_final:\n", E_final)

        # remove excluded vertices from V
    V_final = np.vstack([V[~v_excluded_bool], np.stack(list(V_new.values()))])
    # print("V_final:\n", V_final)

    A_final = np.vstack([A, a_new])
    b_final = np.hstack([b, b_new])

    return V_final, E_final, A_final, b_final

