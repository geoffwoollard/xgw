import ot
import numpy as np
from scipy import sparse


def safe_plot(arr):
    return arr[np.isfinite(arr)]

def gw_matrix(coords_ca_1, coords_ca_2, symmetric, n_skip, flip):

    gw_losses = np.zeros((coords_ca_1.shape[0], coords_ca_2.shape[0]))
    
    for idx_1 in range(coords_ca_1.shape[0]):
        for idx_2 in range(coords_ca_2.shape[0]):
            if idx_1 < idx_2:
                if symmetric: continue
            if flip: 
                coords_ca_2_ = coords_ca_2[idx_2,::n_skip].copy()
                coords_ca_2_[:,0] *= -1
            else:
                coords_ca_2_ = coords_ca_2[idx_2,::n_skip]

            C1 = ot.dist(coords_ca_1[idx_1,::n_skip], coords_ca_1[idx_1,::n_skip])
            C2 = ot.dist(coords_ca_2_, coords_ca_2_)
            p = ot.unif(coords_ca_1[idx_1,::n_skip].shape[0])
            q = ot.unif(coords_ca_2_.shape[0])
            gw_loss = ot.gromov_wasserstein2(C1, C2, p, q)
            gw_losses[idx_1, idx_2] = gw_loss
            if symmetric:
                gw_losses[idx_2, idx_1] = gw_loss
    return gw_losses

def canonicalize_vertices(vertices, decimals=5):
    vertices = np.round(vertices, decimals=decimals)
    vertices = np.unique(vertices, axis=0)
    return vertices

def canonicalize_halfspaces(A, b, decimals=5, tol=1e-12):
    """
    Canonicalize halfspaces A x <= b.

    Steps:
    1. Normalize each row so ||a_i|| = 1
    2. Fix sign ambiguity (make b_i >= 0)
    3. Round for numerical stability
    4. Remove duplicates
    5. Sort rows for deterministic ordering

    Args:
        A: (m, d) array
        b: (m,) array
        decimals: rounding precision
        tol: small threshold to avoid division by zero

    Returns:
        A_canon, b_canon
    """

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    # --- Step 1: normalize rows ---
    norms = np.linalg.norm(A, axis=1)
    mask = norms > tol  # avoid degenerate rows

    A = A[mask]
    b = b[mask]
    norms = norms[mask]

    A = A / norms[:, None]
    b = b / norms

    # --- Step 2: fix sign ambiguity ---
    # enforce b >= 0
    sign_flip = b < 0
    A[sign_flip] *= -1
    b[sign_flip] *= -1

    # --- Step 3: round ---
    A = np.round(A, decimals=decimals)
    b = np.round(b, decimals=decimals)

    # --- Step 4: deduplicate ---
    Ab = np.hstack([A, b[:, None]])
    Ab = np.unique(Ab, axis=0)

    # --- Step 5: sort ---
    # lexicographic sort for determinism
    sort_idx = np.lexsort(Ab.T[::-1])
    Ab = Ab[sort_idx]

    A_canon = Ab[:, :-1]
    b_canon = Ab[:, -1]

    return A_canon, b_canon

def cast_to_dense_if_sparse(matrix):
    if sparse.issparse(matrix):
        return matrix.toarray()
    else:
        return matrix