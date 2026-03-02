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

        for i in range(n):
            for j in range(i + 1, n):
                if BTB[i, j] == self.r - 1:
                    self.D[i, j] = self.D[j, i] = 1

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