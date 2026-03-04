import numpy as np


class ExtremePointPolytope:
    """
    Minimal faithful implementation of Section A.3
    """

    def __init__(self, E, B, dim=3):
        """
        E : (n,3) vertices
        B : (m,n) binary active-constraint matrix
        """
        self.r = dim
        self.E = np.asarray(E, float)
        self.B = np.asarray(B, np.uint8)

        self.n_constraints = self.B.shape[0]

        self.rebuild_adjacency()

    # --------------------------------------------------
    # adjacency from B^T B rule
    # --------------------------------------------------
    def rebuild_adjacency(self):
        n = self.E.shape[0]
        self.D = np.zeros((n, n), dtype=np.uint8)

        BTB = self.B.T @ self.B
        D_vec = BTB - np.diag(BTB.diagonal())
        D_vec = (D_vec == self.r - 1).astype(np.uint8)
        self.D = D_vec

        # for i in range(n):
        #     for j in range(i + 1, n):
        #         if BTB[i, j] == self.r - 1:
        #             self.D[i, j] = self.D[j, i] = 1
        # assert np.array_equal(self.D, D_vec), "Adjacency matrix does not match B^T B rule"

    # --------------------------------------------------
    # add constraint (A_n, b_n)
    # --------------------------------------------------
    def add_constraint(self, a_new, b_new):

        a_new = np.asarray(a_new, float)

        values = self.E @ a_new
        feasible = values <= b_new + 1e-12
        infeasible_idx = np.where(~feasible)[0]

        new_vertices = []
        C_cols = []
        O_links = []

        # ---------- Step B (paper) ----------
        for i in infeasible_idx:

            neighbors = np.where(self.D[i] == 1)[0]

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


class ExtremePointPolytopeSparse:

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------
    def __init__(self, E, masks, A=None, b=None):
        self.E = np.asarray(E, float)
        self.masks = np.array(masks, dtype=object)

        self.A = [] if A is None else list(A)
        self.b = [] if b is None else list(b)

        self.r = 3  # dimension
        self.D = self._build_adjacency()

    # --------------------------------------------------
    # Adjacency helpers
    # --------------------------------------------------
    def _adjacent(self, m1, m2):
        return (m1 & m2).bit_count() == self.r - 1

    def _build_adjacency(self):
        n = len(self.masks)
        D = np.zeros((n, n), dtype=int) # TODO: np.uint8 or sparse?

        for i in range(n):
            for j in range(i + 1, n):
                if self._adjacent(self.masks[i], self.masks[j]):
                    D[i, j] = D[j, i] = 1
        return D

    def _build_adjacency_subset(self, masks):
        n = len(masks)
        D = np.zeros((n, n), dtype=int) # TODO: np.uint8 or sparse?

        for i in range(n):
            for j in range(i + 1, n):
                if (masks[i] & masks[j]).bit_count() == self.r - 1:
                    D[i, j] = D[j, i] = 1
        return D

    # --------------------------------------------------
    # MAIN ALGORITHM (Section A.3)
    # --------------------------------------------------
    def add_constraint(self, a_new, b_new):

        a_new = np.asarray(a_new, float)

        # ---------- Step A: classify vertices ----------
        vals = self.E @ a_new
        feasible = vals <= b_new
        infeasible_idx = np.where(~feasible)[0]

        if len(infeasible_idx) == 0:
            return

        new_vertices = []
        new_masks = []
        O_links = []

        new_bit = 1 << len(self.A)

        # ---------- Step B: generate new vertices ----------
        for i in infeasible_idx:

            neighbors = np.where(self.D[i] == 1)[0]

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

        # ---------- Step C: remove infeasible vertices ----------
        E_old = self.E[feasible]
        masks_old = self.masks[feasible]

        index_map = {
            old_i: new_i
            for new_i, old_i in enumerate(np.where(feasible)[0])
        }

        # ---------- Step D: concatenate extreme points ----------
        P = np.array(new_vertices)

        self.E = np.vstack([E_old, P])
        self.masks = np.concatenate(
            [masks_old, np.array(new_masks, dtype=object)]
        )

        # ---------- Step E: build O matrix ----------
        n_old = len(E_old)
        n_new = len(new_masks)

        O = np.zeros((n_old, n_new), dtype=int) # TODO: np.uint8 or sparse?

        for k, old_j in enumerate(O_links):
            O[index_map[old_j], k] = 1

        # ---------- Step F: compute N block ----------
        N = np.zeros((n_new, n_new), dtype=int)

        for i in range(n_new):
            for j in range(i + 1, n_new):

                shared = (
                    (new_masks[i] & new_masks[j]) & ~new_bit
                ).bit_count()

                if shared == self.r - 2:
                    N[i, j] = N[j, i] = 1

        # ---------- Step G: assemble new adjacency ----------
        D_old = self._build_adjacency_subset(masks_old)

        top = np.hstack([D_old, O])
        bottom = np.hstack([O.T, N])
        self.D = np.vstack([top, bottom])

        # store constraint
        self.A.append(a_new)
        self.b.append(b_new)