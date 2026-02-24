import cvxpy as cp
import numpy as np


class IncrementalQPProjector:
    """
    Solve   minimize ||x - v||^2  subject to Ax <= b,
    and efficiently add new half-space constraints.

    Uses CVXPY + OSQP with warm-starting.
    """
    def __init__(self, dim, A=None, b=None, eps_abs=1e-5, eps_rel=1e-5, max_iter=10000, verbose=False):
        self.dim = dim
        self.x = cp.Variable(dim)
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.max_iter = max_iter
        self.solver_opts = {'eps_abs': self.eps_abs, 'eps_rel': self.eps_rel, 'max_iter': self.max_iter}
        self.verbose = verbose

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
        self.problem.solve(solver=solver, warm_start=warm, verbose=self.verbose, **self.solver_opts)
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


def flat_basis(e_base):
    a,b,c = e_base.shape
    e_base = e_base.reshape((a*b,c))
    return (e_base)


class OptimalProjectedCoupling:
    """
    Solve   minimize ||p(x) - v||^2  subject to Ax <= b, and Cx == d

    Uses CVXPY + OSQP. (maybe use other solver)
    """
    def __init__(self, proj_dim, e_base, mu=None, nu=None, eps_abs=1e-5, eps_rel=1e-5, max_iter=10000, verbose=False):
        l_mu = len(mu)
        l_nu = len(nu)
        dim = l_mu*l_nu
        b = np.zeros(dim)
        C_mu =np.ones(l_mu)
        C_nu = np.ones(l_nu)
        
        self.dim = dim
        self.x = cp.Variable(dim)
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.max_iter = max_iter
        self.solver_opts = {'eps_abs': self.eps_abs, 'eps_rel': self.eps_rel, 'max_iter': self.max_iter}
        self.verbose = verbose
        

        # initially possibly empty constraint list
        self.constraints = []
        
        self.constraints.append( self.x >= b)
        x_reshape = cp.reshape(self.x, shape=(l_mu,l_nu), order='C')
        self.constraints.append(C_mu @ x_reshape -mu == 0)
        self.constraints.append( x_reshape @ C_nu -nu == 0)

        # v is a parameter so we don't have to rebuild the problem
        self.v = cp.Parameter(proj_dim)

        # objective = ||p(x) - v||^2
        e_base = flat_basis(e_base) # flattening the base for the projection
        self.obj = cp.Minimize(cp.sum_squares(self.x@e_base - self.v))

        # build the problem
        self.problem = cp.Problem(self.obj, self.constraints)


    def solve(self, v_value, solver=cp.OSQP):
        """
        Solve the QP for given v.
        """
        self.v.value = v_value
        self.problem.solve(solver=solver, verbose=self.verbose, **self.solver_opts)
        return self.x.value, self.problem.value
    
    