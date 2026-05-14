from math import log2
import numpy as np
from scipy import sparse
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_neighbours(D, idx):
    if isinstance(D, np.ndarray):
        neighbors = np.where(D[idx] == 1)[0]
    elif sparse.issparse(D):
        neighbors = D[idx].nonzero()[1]
    else:
        raise ValueError("Adjacency matrix D is unrecognized type: {}".format(type(D)))
    return neighbors

class ExtremePointPolytope:
    """
    Minimal faithful implementation of Section A.3
    """

    def __init__(self, E, B, add_constraint_tol=1e-17, D_chunk_size=5000, use_D_sparse=True):
        """
        E : (n,dim) vertices
        B : (m,n) binary active-constraint matrix
        """
        self.r = E.shape[1]
        self.E = np.asarray(E, float)
        self.B = np.asarray(B, np.uint8)

        self.n_constraints = self.B.shape[0]
        self.add_constraint_tol = add_constraint_tol
        self.D_chunk_size = D_chunk_size

        self.use_D_sparse = use_D_sparse
        self.rebuild_adjacency()

    # --------------------------------------------------
    # adjacency from B^T B rule
    # --------------------------------------------------

    def rebuild_adjacency(self):
        if self.use_D_sparse:
            self.rebuild_adjacency_sparse_chunks()
        else:
            self.rebuild_adjacency_dense()
        
    def rebuild_adjacency_dense(self):
        n = self.E.shape[0]
        self.D = np.zeros((n, n), dtype=np.uint8)

        BTB = self.B.T @ self.B
        D_vec = BTB - np.diag(BTB.diagonal())
        D_vec = (D_vec >= self.r - 1).astype(np.uint8) # BEWARE MARJOR BUG: adjacent if they share at least r-1 active constraints (not exact!)
        self.D = D_vec

    def rebuild_adjacency_sparse_chunks(self):
        """Compute adjacency matrix D in chunks and store as sparse CSR (bool)."""
        n = self.E.shape[0]
        chunk_size = self.D_chunk_size
        
        rows, cols = [], []
        
        for i in range(0, n, chunk_size):
            i_end = min(i + chunk_size, n)
            
            # compute B^T B for this chunk of columns
            B_chunk = self.B[:, i:i_end]  # (m, chunk_size)
            BTB_chunk = self.B.T @ B_chunk  # (n, chunk_size)
            
            # find all (i,j) where BTB_chunk[i,j] >= r-1
            i_idx, local_j_idx = np.where(BTB_chunk >= self.r - 1)
            
            # convert local column indices to global
            global_j_idx = local_j_idx + i
            
            # keep only upper triangle (i < j)
            mask = i_idx < global_j_idx
            rows.extend(i_idx[mask])
            cols.extend(global_j_idx[mask])
        
        # build upper triangle COO with bool dtype
        D_upper = sparse.coo_matrix(
            (np.ones(len(rows), dtype=bool), (rows, cols)),
            shape=(n, n),
            dtype=bool
        )
        
        # symmetrize: D = D_upper + D_upper^T
        self.D = D_upper + D_upper.T
        self.D = self.D.tocsr()  # convert to CSR for efficient storage/access


    # --------------------------------------------------
    # add constraint (A_n, b_n)
    # --------------------------------------------------
    def add_constraint(self, a_new, b_new):

        a_new = np.asarray(a_new, float)

        values = self.E @ a_new
        feasible = values <= b_new + self.add_constraint_tol
        infeasible_idx = np.where(~feasible)[0]

        new_vertices = []
        C_cols = []
        O_links = []

        # ---------- Step B (paper) ----------
        for i in infeasible_idx:

            neighbors = get_neighbours(self.D, i)

            for j in neighbors:
                if not feasible[j]:
                    continue

                vi = self.E[i]
                vj = self.E[j]

                direction = vj - vi
                denom = direction @ a_new
                if abs(denom) < 1e-12:
                    continue

                t = (b_new - vi @ a_new) / denom
                if not (0 <= t <= 1):
                    continue

                p = vi + t * direction
                new_vertices.append(p)

                # ---- construct C column ----
                # C[:,k] = B[:,i] ⊙ B[:,j]
                inherited = self.B[:, i] & self.B[:, j]
                C_cols.append(inherited)

                O_links.append(j)

        if len(new_vertices) == 0:
            return

        P = np.vstack(new_vertices)
        C = np.column_stack(C_cols).astype(np.uint8)

        self.C = C   # store for inspection/debugging

        # ---------- Step C: remove infeasible ----------
        E_old = self.E[feasible]
        B_old = self.B[:, feasible]

        # ---------- Step D: extend B ----------
        # Bn = [ B  C
        #        0  1 ]

        zeros = np.zeros((1, B_old.shape[1]), dtype=np.uint8)
        ones = np.ones((1, C.shape[1]), dtype=np.uint8)

        B_top = np.hstack([B_old, C])
        B_bottom = np.hstack([zeros, ones])

        self.B = np.vstack([B_top, B_bottom])

        self.n_constraints += 1

        # ---------- Step E: update vertices ----------
        self.E = np.vstack([E_old, P])

        # ---------- Step F: adjacency ----------
        self.rebuild_adjacency()


def masks_from_B(B):
    """B: (m_constraints, n_vertices) boolean array -> list/array of Python int masks"""
    m, n = B.shape
    masks = []
    for j in range(n):
        mask = 0
        col = B[:, j]
        for i, val in enumerate(col):
            if val:
                mask |= (1 << i)
        masks.append(mask)
    return np.array(masks, dtype=object)


def masks_from_B_vectorized(B):
    m, n = B.shape
    bits = np.array([1 << i for i in range(m)], dtype=object)   # Python ints
    # B.T is (n, m); multiply in object dtype then sum along axis=1
    return (B.T.astype(object) * bits).sum(axis=1)

class ExtremePointPolytopeSparse:

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------
    def __init__(self, E, masks, A=None, b=None, D_chunk_size=30_000, use_D_sparse=True):
        self.E = np.asarray(E, float)
        self.masks = np.array(masks, dtype=object)

        self.A = [] if A is None else list(A)
        self.b = [] if b is None else list(b)
        self.D_chunk_size = D_chunk_size

        self.r = E.shape[1]
        self.D = self._build_adjacency()
        self.use_D_sparse = use_D_sparse
        if use_D_sparse:
            self.D = sparse.csr_matrix(self.D, dtype=bool)
        self._POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


    # --------------------------------------------------
    # Adjacency helpers
    # --------------------------------------------------
    def _adjacent(self, m1, m2):
        return (m1 & m2).bit_count() >= self.r - 1

    def _build_adjacency(self):
        n = len(self.masks)
        D = np.zeros((n, n), dtype=bool) # TODO: np.uint8 or sparse?

        for i in range(n):
            for j in range(i + 1, n):
                if self._adjacent(self.masks[i], self.masks[j]):
                    D[i, j] = D[j, i] = True
        return D

    def _build_adjacency_subset(self, masks, use_sparse):
        if use_sparse:
            if max(masks) < 2**64:
                return self._build_adjacency_subset_vectorized_chunked(masks, self.D_chunk_size)
            else:
                logger.warning("Masks exceed 64 bits, falling back to non-vectorized sparse chunked adjacency.")
                return self._build_adjacency_subset_sparse_chunks(masks, self.D_chunk_size)  
        else:
            return self._build_adjacency_subset_dense(masks)

    def _build_adjacency_subset_dense(self, masks):
        n = len(masks)
        D = np.zeros((n, n), dtype=bool) # TODO: np.uint8 or sparse?

        for i in range(n):
            for j in range(i + 1, n):
                if (masks[i] & masks[j]).bit_count() >= self.r - 1:
                    D[i, j] = D[j, i] = True
        return D
    
    def _build_adjacency_subset_sparse_chunks(self, masks, chunk_size):
        """Build sparse adjacency for a subset of masks (chunked)."""
        n = len(masks)
        # chunk_size = self.D_chunk_size
        
        rows, cols = [], []
        
        for j_start in range(0, n, chunk_size):
            j_end = min(j_start + chunk_size, n)
            logger.info(f'Building adjacency for columns {j_start} to {j_end} (total {n})')
            
            for local_j, j in enumerate(range(j_start, j_end)):
                m_j = masks[j]
                
                for i in range(j):
                    m_i = masks[i]
                    shared = (m_i & m_j).bit_count()
                    
                    if shared >= self.r - 1:
                        rows.append(i)
                        cols.append(j)

        logger.info(f'Found {len(rows)} adjacencies (upper triangle). Building sparse COO.')
        D_upper = sparse.coo_matrix(
            (np.ones(len(rows), dtype=bool), (rows, cols)),
            shape=(n, n),
            dtype=bool
        )
        logger.info('Symmetrize: D = D_upper + D_upper^T')
        D = D_upper + D_upper.T
        logger.info(f'Convert to CSR format')
        D_csr = D.tocsr()
        logger.info(f'Adjacency subset built: shape={D_csr.shape}, nnz={D_csr.nnz}')
        return D_csr

    def _build_adjacency_subset_vectorized_chunked(self, masks, chunk_size=None):
        """Build adjacency matrix in chunks, assemble as sparse CSR."""
        if chunk_size is None:
            chunk_size = self.D_chunk_size
        
        n = len(masks)
        rows, cols = [], []
        
        # Determine dtype for masks
        max_mask = max(masks)
        if max_mask < 2**64:
            masks = np.asarray(masks, dtype=np.uint64)
        else:
            masks = np.asarray(masks, dtype=object)
        
        logger.info(f'Process chunks of columns. log_2 max_mask={log2(max_mask)}, masks.shape={masks.shape}, dtype={masks.dtype}')
        for j_start in range(0, n, chunk_size):
            j_end = min(j_start + chunk_size, n)
            logger.info(f'Processing columns {j_start} to {j_end}')
            masks_j_chunk = masks[j_start:j_end]
            
            # pairwise bitwise AND: all rows vs chunk of columns
            # shape: (n, len(chunk))
            logger.info(f'Computing pairwise bitwise AND for chunk. masks shape: {masks.shape}, chunk shape: {masks_j_chunk.shape}')
            inter = masks[:, None] & masks_j_chunk[None, :]
            
            # vectorized popcount
            logger.info(f'Computing bit counts for chunk. inter shape: {inter.shape}')
            bitcounts = np.bitwise_count(inter)
            
            logger.info(f'Found adjacencies (excluding diagonal).')
            i_idx, local_j_idx = np.where(bitcounts >= self.r - 1)
            global_j_idx = local_j_idx + j_start
            
            logger.info(f'Keeping only upper triangle (i < j).')
            mask = i_idx < global_j_idx
            logger.info('Update rows and cols for sparse COO construction.')
            rows.extend(i_idx[mask])
            cols.extend(global_j_idx[mask])
        
        logger.info('Building upper triangle COO with bool dtype.')
        D_upper = sparse.coo_matrix(
            (np.ones(len(rows), dtype=bool), (rows, cols)),
            shape=(n, n),
            dtype=bool
        )
        
        logger.info('Symmetrize: D = D_upper + D_upper^T')
        D = D_upper + D_upper.T
        logger.info(f'Convert to CSR format')
        D_csr = D.tocsr()
        logger.info(f'Adjacency subset built: shape={D_csr.shape}, nnz={D_csr.nnz}')
        return D_csr

    def _build_adjacency_subset_vectorized(self, masks):
        if max(masks) < 2**64:
            masks = np.asarray(masks, dtype=np.uint64)
        else:
            masks = np.asarray(masks, dtype=object)
        
        # pairwise bitwise AND
        inter = masks[:, None] & masks[None, :]

        # vectorized popcount
        bitcounts = np.bitwise_count(inter)

        D = (bitcounts >= self.r - 1).astype(bool)

        np.fill_diagonal(D, False)
        return D
    

    def _build_adjacency_subset_popcount_table(self, masks):
        masks = np.asarray(masks, dtype=np.uint64)
        
        xor_matrix = masks[:, None] ^ masks[None, :]
        xor_matrix = np.ascontiguousarray(xor_matrix, dtype=np.uint64)
        print(f"XOR matrix computed: {xor_matrix}")
        
        # Correct 3D byte view
        bytes_view = xor_matrix.view(np.uint8).reshape(xor_matrix.shape + (8,))
        
        # Hamming distance
        bitcounts = self._POPCOUNT_TABLE[bytes_view].sum(axis=-1)
        
        # adjacency: 2 or more bits differ
        D = (bitcounts >= self.r - 1).astype(bool)
        np.fill_diagonal(D, False)
        
        return D
 

    # --------------------------------------------------
    # MAIN ALGORITHM (Section A.3)
    # --------------------------------------------------
    def add_constraint(self, a_new, b_new):


        a_new = np.asarray(a_new, float)

        logger.info('# ---------- Step A: classify vertices ----------')
        vals = self.E @ a_new
        feasible = vals <= b_new
        infeasible_idx = np.where(~feasible)[0]

        if len(infeasible_idx) == 0:
            return

        new_vertices = []
        new_masks = []
        O_links = []

        new_bit = 1 << len(self.A)

        logger.info('# ---------- Step B: generate new vertices ----------')
        for i in infeasible_idx:

            neighbors = get_neighbours(self.D, i)

            for j in neighbors:
                if not feasible[j]:
                    continue

                vi = self.E[i]
                vj = self.E[j]

                direction = vj - vi
                denom = direction @ a_new
                if abs(denom) < 1e-12:
                    continue

                t = (b_new - vi @ a_new) / denom
                p = vi + t * direction

                new_vertices.append(p)

                # C construction (bitwise AND)
                inherited = self.masks[i] & self.masks[j]
                new_masks.append(inherited | new_bit)

                O_links.append(j)

        if len(new_vertices) == 0:
            return

        logger.info('# ---------- Step C: remove infeasible vertices ----------')
        E_old = self.E[feasible]
        masks_old = self.masks[feasible]

        index_map = {
            old_i: new_i
            for new_i, old_i in enumerate(np.where(feasible)[0])
        }

        logger.info('# ---------- Step D: concatenate extreme points ----------')
        P = np.array(new_vertices)

        self.E = np.vstack([E_old, P])
        self.masks = np.concatenate(
            [masks_old, np.array(new_masks, dtype=object)]
        )

        logger.info('# ---------- Step E: build O matrix ----------')
        n_old = len(E_old)
        n_new = len(new_masks)
        
        def build_O(n_old, n_new, O_links, index_map, use_sparse=True):
            """Build O matrix (n_old × n_new) efficiently.
            
            O[i, k] = 1 iff old vertex i is linked to new vertex k.
            Typically very sparse (one 1 per column).
            """
            if use_sparse:
                # Build sparse COO directly: only store the 1s
                rows = []
                cols = []
                for k, old_j in enumerate(O_links):
                    rows.append(index_map[old_j])
                    cols.append(k)
                
                O = sparse.coo_matrix(
                    (np.ones(len(rows), dtype=bool), (rows, cols)),
                    shape=(n_old, n_new),
                    dtype=bool
                )
                return O.tocsr()  # CSR for efficient row/col slicing
            else:
                # Dense fallback (for small polytopes)
                O = np.zeros((n_old, n_new), dtype=bool)
                for k, old_j in enumerate(O_links):
                    O[index_map[old_j], k] = True
                return O
            
        O = build_O(n_old, n_new, O_links, index_map, use_sparse=self.use_D_sparse)

        
        logger.info('# ---------- Step F: compute N block ----------')      
        def build_N(n_new, new_masks, new_bit, r, use_sparse=True):
            """Build N matrix (n_new × n_new) efficiently.
            
            N[i, j] = 1 iff new vertices i and j share >= r-2 constraints
            (excluding the new constraint).
            """
            if use_sparse:
                rows, cols = [], []
                
                for i in range(n_new):
                    for j in range(i + 1, n_new):
                        shared = (
                            (new_masks[i] & new_masks[j]) & ~new_bit
                        ).bit_count()
                        
                        if shared >= r - 2:
                            rows.append(i)
                            cols.append(j)
                
                # build upper triangle COO with bool dtype
                N = sparse.coo_matrix(
                    (np.ones(len(rows), dtype=bool), (rows, cols)),
                    shape=(n_new, n_new),
                    dtype=bool
                )
                # symmetrize
                N = N + N.T
                return N.tocsr()
            else:
                # Dense fallback
                N = np.zeros((n_new, n_new), dtype=bool)
                
                for i in range(n_new):
                    for j in range(i + 1, n_new):
                        shared = (
                            (new_masks[i] & new_masks[j]) & ~new_bit
                        ).bit_count()
                        
                        if shared >= self.r - 2:
                            N[i, j] = N[j, i] = True
                return N
        

        N = build_N(n_new, new_masks, new_bit, self.r, use_sparse=self.use_D_sparse)


        logger.info('# ---------- Step G: assemble new adjacency ----------')
        D_old = self._build_adjacency_subset(masks_old, use_sparse=self.use_D_sparse)
        # D_old_also_working = self._build_adjacency_subset_vectorized(masks_old)
        # assert np.array_equal(D_old_working.toarray(), D_old_also_working), "Adjacency subsets do not match!"
        # D_old = self._build_adjacency_subset_vectorized_chunked(masks_old, self.D_chunk_size)
        # assert np.array_equal(D_old_working.toarray(), D_old.toarray()), "Adjacency subsets do not match!"

        logger.info('# ---------- Step H: assemble final D ----------')
        def assemble_D(use_D_sparse, D_old, O, N):
            if use_D_sparse:
                D_updated = sparse.bmat([
                    [D_old, O],
                    [O.T, N]
                ], format='csr', dtype=bool)
            else:
                top = np.hstack([D_old, O])
                bottom = np.hstack([O.T, N])
                D_updated = np.vstack([top, bottom])
            return D_updated
        
        self.D = assemble_D(self.use_D_sparse, D_old, O, N)
        # D_dense = assemble_D(False, 
        #                      self._build_adjacency_subset(masks_old, use_sparse=False), 
        #                      build_O(n_old, n_new, O_links, index_map, use_sparse=False),
        #                     build_N(n_new, new_masks, new_bit, self.r, use_sparse=False))
        # assert np.array_equal(self.D.toarray(), D_dense), "Sparse and dense D do not match!"

        logger.info('# ---------- Step I: store constraint ----------')
        self.A.append(a_new)
        self.b.append(b_new)

# # def build_popcount_table():
# global _POPCOUNT_TABLE
# _POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)



if __name__ == "__main__":
    # square
    V = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    d = 2
    assert V.shape == (4,d)
    assert A.shape == (4,d)
    b = np.array([1, 1, 0, 0])
    from xgw.hyperplane_approx import build_B_from_H_and_V
    B = build_B_from_H_and_V(A, b, V,)
    masks = masks_from_B(B)
    poly = ExtremePointPolytopeSparse(V, masks, A, b)
    print(poly.E)
    print(poly.D)
    delta = 0.1
    poly.add_constraint(np.array([1, 1]), d - delta)
    print(poly.E)
    print(poly.D)
    D = poly._build_adjacency_subset_popcount_table(poly.masks)
    print(D)
#     masks = np.array([0b0111, 0b1011, 0b1101, 0b1110,], dtype=np.uint64)
#     D = build_adjacency_subset(masks, 3)
#     print(D)

