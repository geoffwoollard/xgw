import cvxpy as cp
import numpy as np


class IncrementalQPProjector:
    """
    Solve   minimize ||x - v||^2  subject to Ax <= b,
    and efficiently add new half-space constraints.

    Uses CVXPY + OSQP with warm-starting.
    """
    def __init__(self, dim, A=None, b=None):
        self.dim = dim
        self.x = cp.Variable(dim)

        # initially possibly empty constraint list
        self.constraints = []
        if A is not None and b is not None:
            self.constraints.append(A @ self.x <= b)

        # v is a parameter so we don't have to rebuild the problem
        self.v = cp.Parameter(dim)

        # objective = ||x - v||^2
        self.obj = cp.Minimize(cp.sum_squares(self.x - self.v))

        # build the problem
        self.problem = cp.Problem(self.obj, self.constraints)

    def solve(self, v_value, warm=True, solver=cp.OSQP):
        """
        Solve the QP for given v.
        warm = True uses warm_start=True in CVXPY.
        """
        self.v.value = v_value
        self.problem.solve(solver=solver, warm_start=warm, verbose=False)
        return self.x.value, self.problem.value

    def add_constraint(self, a_new, b_new):
        """
        Add a single new constraint a_new^T x <= b_new.
        Does NOT solve; just updates the problem.
        """
        a_new = np.asarray(a_new)
        b_new = float(b_new)
        cons = (a_new @ self.x <= b_new)
        self.constraints.append(cons)
        self.problem = cp.Problem(self.obj, self.constraints)

    def solve_with_new_constraint(self, v_value, a_new, b_new, warm=True):
        """
        Convenience: add a constraint and solve immediately (warm-start).
        """
        self.add_constraint(a_new, b_new)
        return self.solve(v_value, warm=warm)
