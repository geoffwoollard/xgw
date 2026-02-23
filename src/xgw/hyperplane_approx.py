import numpy as np
from numpy.linalg import qr
from numba import njit
import ot
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from pypoman import compute_polytope_halfspaces, compute_polytope_vertices
except ImportError as e:
    logger.info("pypoman is required for Hyperplane_approx module. Please install it via pip: pip install pypoman")


def _iteration_loop_hausdorff(mu, nu, p_plus, p_minus, e_base, previous_solutions_to_reuse, emd_kwargs):
    x_0, v_0, objective, previous_solutions_to_reuse = hausdorff(p_plus, p_minus, previous_solutions_to_reuse)
    g = new_direction(x_0, v_0, p_minus)
    g_hat, g_star = compute_hyperplane(mu, nu, g, e_base, emd_kwargs)
    p_plus, p_minus = update_box(p_plus, p_minus, [[g, g_hat]], [g_star])
    return p_plus, p_minus, objective, previous_solutions_to_reuse, x_0, v_0


def _run_approx(mu, nu, space_x, space_y, emd_kwargs, niter=100, epsilon=1e-15):
    e_base, R, p_plus, p_minus = initial_box(space_x, space_y, mu, nu, emd_kwargs)
    logger.info('box initialized')
    
    objective_list = []
    x_0_list, v_0_list = [], []
    for iter in range(niter):
        previous_solutions_to_reuse = {} # todo: fix bug with reusing previous solutions
        logger.info(f'Iteration {iter}')
        p_plus, p_minus, objective, _, x_0, v_0 = _iteration_loop_hausdorff(mu, nu, p_plus, p_minus, e_base, previous_solutions_to_reuse, emd_kwargs)
        objective_list.append(objective)
        x_0_list.append(x_0)
        v_0_list.append(v_0)
        logger.info(f'Iteration {iter}, Hausdorff distance: {objective}')
        if objective < epsilon:
            break
    return p_plus, p_minus, objective, previous_solutions_to_reuse, objective_list, x_0_list, v_0_list
        

def iteration_loop_hausdorff(mu, nu, p_plus, p_minus, e_base, previous_solutions_to_reuse, emd_kwargs):
    x_0, v_0, objective, previous_solutions_to_reuse = hausdorff(p_plus, p_minus, previous_solutions_to_reuse)
    g = new_direction(x_0, v_0, p_minus)
    g_hat, g_star = compute_hyperplane(mu, nu, g, e_base, emd_kwargs)
    p_plus, p_minus = update_box(p_plus, p_minus, [[g, g_hat]], [g_star])
    return p_plus, p_minus, objective, previous_solutions_to_reuse


def run_approx(mu, nu, space_x, space_y, emd_kwargs, niter=100, epsilon=1e-15):
    e_base, R, p_plus, p_minus = initial_box(space_x, space_y, mu, nu, emd_kwargs)
    logger.info('box initialized')
    
    objective_list = []
    for iter in range(niter):
        previous_solutions_to_reuse = {} # todo: fix bug with reusing previous solutions
        logger.info(f'Iteration {iter}')
        p_plus, p_minus, objective, previous_solutions_to_reuse  = iteration_loop_hausdorff(mu, nu, p_plus, p_minus, e_base, previous_solutions_to_reuse, emd_kwargs)
        objective_list.append(objective)
        logger.info(f'Iteration {iter}, hausdorff distance: {objective}')
        if objective < epsilon:
            break
    return p_minus, objective, R, e_base


def construct_basis_eij(space_x, space_y):
    # finding orthonormal basis : Q = [e_1,...,e_dx*dy] orthonormal base (ei flatten), f_base = [f1,...,f_dx*dy] = e_base@R, R triangular superior matrix of size dx*dy^2, e_base (N, M) by dx*dy tensor, f_base (N, M) by dx*dy tensor
    N, dx = space_x.shape
    M, dy = space_y.shape
    stacked_base = np.zeros((N*M, dx*dy))
    assert dx == dy
    for i in range(dx):
        for j in range(dy):
            stacked_base[:, i+j*dx] = np.ravel(np.outer(space_x[:,i], space_y[:,j])) # flattening vectors to use QR decomposition
    Q, R = qr(stacked_base, mode='reduced') 
    assert np.all(np.diag(R)!=0), f"at least one marginal is supported on a d-1 vector space" 
    e_base = np.reshape(Q,(N, M, dx*dy)) # reshape to  physical dimensions, now we have a N by M by dx*dy tensor
    return e_base, R


@njit
def e_to_f(vect, R, d):
    sol = (vect@R).reshape((d,d))
    return sol


@njit
def f_to_e(vect, R_inv):
    return np.ravel(vect) @ R_inv


class DoubleDescription():
    def __init__(self, duplicate_tol=1e-5, implementation='cdd'):
        self.V = []
        self.H = ()
        self.duplicate_tol = duplicate_tol
        self.implementation = implementation
        # self.n_decimals_for_v_round = 30

    def remove_duplicates_V(self):
        """
        Remove duplicates from self.V up to a Euclidean distance tolerance.
        Uses O(n log n) quantization + np.unique instead of O(n^2) pairwise checks.
        """
        V_array = np.asarray(self.V, dtype=float)
        tol = float(self.duplicate_tol)

        if len(V_array) == 0:
            return

        # Quantize vertices to tolerance grid
        scale = 1.0 / tol
        V_quant = np.round(V_array * scale).astype(np.int64)

        # Find unique rows (lexicographic)
        _, unique_indices = np.unique(V_quant, axis=0, return_index=True)

        # Preserve original order
        unique_indices.sort()

        logger.info(f"Original number of vertices: {len(self.V)}")
        self.V = [V_array[i] for i in unique_indices]
        logger.info(f"Removed duplicates, new number of vertices: {len(self.V)}")

    def check_feasibility_V(self, A, b):
        import numpy as np
        from scipy.optimize import linprog

        # A x <= b
        c = np.zeros(A.shape[1])
        res = linprog(c, A_ub=A, b_ub=b)
        logger.info(f'Feasibility check result: {res.success}')
        logger.info(f'Linprog result: {res}')

    def H_to_V(self):
        if self.implementation == 'h_to_v_edges':
            raise NotImplementedError('h_to_v_edges implementation is not implemented yet')
        elif self.implementation == 'cdd':
            A, b = self.H
            logger.info(f'Computing vertices from half-planes: A shape {A.shape}, b shape {b.shape}')
            logger.info(f'Half-planes: {A}, {b}')
            assert not self.check_feasibility_V(A, b)
            
            def stabilize_compute_polytope_vertices(A, b, decimals_start=15, decimals_end=3):
                '''Try to compute vertices from halfspaces with decreasing rounding precision to enhance numerical stability.
                
                This function avoids the error Error: Numerical inconsistency is found.  Use the GMP exact arithmetic.
                This error is related to how cdd casts floating point numbers to rationals internally, which can lead to numerical issues.
                In general, rounding more allows the function to succeed, but too much rounding can distort the polytope shape.
                Also, aggresive rounding can fail, while moderate rounding with tiny jitter can succeed.
                Hence the strategy of starting from high precision and decreasing it, trying jitter if needed.
                The jitter is scaled to be small compared to the rounding scale (rounding scale is 1/2 std of the jitter, per component).
                
                '''
                try:
                    V = compute_polytope_vertices(A, b)
                except:
                    H = np.hstack([A, b.reshape(-1, 1)]) # cdd format
                    for decimals in range(decimals_start, decimals_end, -1):
                        V=[]
                        try:
                            # 1. Round vertices to reduce numerical noise
                            H_rounded = [np.round(h, decimals=decimals) for h in H]

                            # 2. Keep only unique vertices
                            H_rounded_unique = np.unique(H_rounded, axis=0)
                            A_rounded_unique = H_rounded_unique[:,:-1]
                            b_rounded_unique = H_rounded_unique[:,-1]

                            # 3. Try compute_polytope_halfspaces
                            try:
                                V = compute_polytope_vertices(A_rounded_unique, b_rounded_unique)
                            except RuntimeError as e:
                                # 4. If it fails due to numerical issues, add tiny jitter
                                scale = 0.5 * 10**(-decimals)
                                jitter = scale * np.random.randn(*H_rounded_unique.shape)
                                H_perturbed = H_rounded_unique + jitter
                                A_perturbed = H_perturbed[:,:-1]
                                b_perturbed = H_perturbed[:,-1]
                                V = compute_polytope_vertices(A_perturbed, b_perturbed)

                        except Exception as e:
                            # print(f'Failed with rounding to {decimals} decimals: {e}')
                            continue  # try next lower precision

                        else:
                            if len(V)>1:
                                # print(f"Success with {decimals} decimals!")
                                break
                return V
            logger.info(f'A,b: {A, b}')
            self.V = stabilize_compute_polytope_vertices(A, b)
            self.remove_duplicates_V()

    def V_to_H(self):
        # loop over high to low decimals to ensure numerical stability. take largest that works

        def stabilize_compute_polytope_halfspaces(V, decimals_start=15, decimals_end=3):
            '''Try to compute halfspaces from vertices with decreasing rounding precision to enhance numerical stability.
            
            This function avoids the error Error: Numerical inconsistency is found.  Use the GMP exact arithmetic.
            This error is related to how cdd casts floating point numbers to rationals internally, which can lead to numerical issues.
            In general, rounding more allows the function to succeed, but too much rounding can distort the polytope shape.
            Also, aggresive rounding can fail, while moderate rounding with tiny jitter can succeed.
            Hence the strategy of starting from high precision and decreasing it, trying jitter if needed.
            The jitter is scaled to be small compared to the rounding scale (rounding scale is 1/2 std of the jitter, per component).
            
            '''
            try:
                A, b = compute_polytope_halfspaces(V)
            except:
                for decimals in range(decimals_start, decimals_end, -1):
                    A=[]
                    try:
                        # 1. Round vertices to reduce numerical noise
                        V_rounded = [np.round(v, decimals=decimals) for v in V]

                        # 2. Keep only unique vertices
                        V_rounded_unique = np.unique(V_rounded, axis=0)

                        # 3. Try compute_polytope_halfspaces
                        try:
                            A, b = compute_polytope_halfspaces(V_rounded_unique)
                        except RuntimeError as e:
                            # 4. If it fails due to numerical issues, add tiny jitter
                            scale = 0.5 * 10**(-decimals)
                            jitter = scale * np.random.randn(*V_rounded_unique.shape)
                            V_perturbed = V_rounded_unique + jitter
                            A, b = compute_polytope_halfspaces(V_perturbed)

                    except Exception as e:
                        # print(f'Failed with rounding to {decimals} decimals: {e}')
                        continue  # try next lower precision

                    else:
                        if len(A)>1:
                            # print(f"Success with {decimals} decimals!")
                            break
                   
            return A, b
        A, b = stabilize_compute_polytope_halfspaces(self.V)
        a_norm = np.linalg.norm(A, axis=1)
        b /= a_norm
        A /= a_norm[:, np.newaxis]
        self.H = (A,b)
        
    def __len__(self):
        # number of vertices and constraints
        return len(self.V), self.H[0].shape 
    
    def __str__(self):
        return f"Vertices: {self.V}\nHalf-planes: {self.H}"
    
    # could be optimized for a family of vertices
    def add_V(self, vertex_list):
        # vertex is a d^2 by 1 vector
        self.V.extend(vertex_list)
        self.remove_duplicates_V()
        self.V_to_H()
    
    # could be optimized for a family of vectors and scalars
    def add_H(self, constraint_list):
        vector_list = [np.transpose(elem[0]) for elem in constraint_list]
        scalar_list = [np.transpose(elem[1]) for elem in constraint_list]
        # vector is a d^2 by 1 vector
        if self.H != ():
            vector_list.append(self.H[0])
            scalar_list.append(self.H[1])
        A = np.vstack(vector_list)
        b = np.hstack(scalar_list)
        self.H = [A, b]
        self.H_to_V()
    
    def get_centroid(self):
        return np.mean(np.array(self.V), axis=0)


def initial_box(space_x, space_y, mu, nu, emd_kwargs):
    '''
    Docstring for initial_box
    
    :param e: orthonormal basis for V
    :param mu: marginal 1
    :param nu: marginal 2
    creates an initial rectangle bounding 
    '''
    e_base, R = construct_basis_eij(space_x, space_y)
    p_plus, p_minus = DoubleDescription(), DoubleDescription()
    vertex_list = []
    half_plans_list = []
    a,b,c = e_base.shape
    for i in range(c):
        for sigma in [-1,1]:
            e_i = np.zeros(c)
            e_i[i] = 1
            g_hat, g_star = compute_hyperplane(mu, nu, sigma*e_i, e_base, emd_kwargs)
            vertex_list.append(g_star)
            half_plans_list.append([sigma*e_i, g_hat])
    update_box(p_plus, p_minus, half_plans_list, vertex_list)
            
    return e_base, R, p_plus, p_minus


def compute_hyperplane(mu, nu, g, e_base, emd_kwargs):
    cost_matrix = function_to_cost(g, e_base)
    map, log = ot.emd(mu, nu, M=-cost_matrix, log=True, **emd_kwargs)
    return -log['cost'], projection(map, e_base)


def projection(pi, e_base):
    logger.info(f'e_base {e_base.shape}, pi {pi.shape}')
    return np.einsum('ijk,ij->k', e_base, pi).reshape(-1,)


def function_to_cost(g, e_base):
    return np.einsum('ijk,k->ij', e_base, g)


def update_box(p_plus, p_minus, half_planes_list, vertex_list):
    logger.info('Adding new half-planes and vertices to the bounding boxes')
    p_plus.add_H(half_planes_list)
    logger.info('Added half-planes to p_plus')
    p_minus.add_V(vertex_list)
    logger.info('Added vertices to p_minus')
    p_minus.V_to_H()
    logger.info('Updated half-planes of p_minus from vertices')
    p_plus.H_to_V()
    logger.info('Updated vertices of p_plus from half-planes')
    return p_plus, p_minus


def new_direction(x_0, v_0, p_minus):
    if np.allclose(x_0, v_0):
        sol = v_0 - p_minus.get_centroid()
        assert np.all(sol != 0), f' zero direction'
        return v_0 - p_minus.get_centroid()
    else:
        sol = v_0-x_0
    return (sol)/np.linalg.norm(sol)


#we can  probably  use numba here, else it may be slow, not sure how numba works with classes though
def hausdorff(p_plus, p_minus, previous_solutions_to_reuse):
    cost = -np.inf
    for vertex in p_plus.V:
        x, objective, _ = solve_dist(vertex, p_minus, previous_solutions_to_reuse)
        # x, objective = solve_dist_brute_force(vertex, p_minus)
        if objective > cost:
            x0, v_0 = x, vertex
            cost = objective
    logger.info(f'Hausdorff distance :{objective}')
    return x0, v_0, objective, previous_solutions_to_reuse


            
def p_plus_outside_p_minus(p_plus, p_minus):
    '''Check p_minus is included in p_plus'''
    A,b = p_plus.H
    residuals = [A@elem - b for elem in p_minus.V]
    residuals = np.array(residuals)
    return residuals        
        
        
def build_new_constraint(A_all, b_all, A_old, b_old):
    # Stack A and b together for comparison
    all_rows = np.hstack([A_all, b_all.reshape(-1,1)])
    old_rows = np.hstack([A_old, b_old.reshape(-1,1)])

    # Find index where row is in all_rows but not in old_rows
    for i, row in enumerate(all_rows):
        if not any(np.all(row == r) for r in old_rows):
            new_index = i
            break
    a_new = A_all[new_index]
    b_new = b_all[new_index]
    return a_new, b_new

def solve_dist(vertex, p_minus, previous_solutions_to_reuse):
    from .qp_incremental_projector import IncrementalQPProjector
    if vertex.tobytes() not in previous_solutions_to_reuse:
        A_all, b_all = p_minus.H
        qp_solver = IncrementalQPProjector(dim=len(vertex), A=A_all, b=b_all)
        x, objective = qp_solver.solve(vertex)
    else:
        qp_solver = previous_solutions_to_reuse[vertex.tobytes()]['solver']
        A_old, b_old = previous_solutions_to_reuse[vertex.tobytes()]['H']
        A_all, b_all = p_minus.H
        a_new, b_new = build_new_constraint(A_all, b_all, A_old, b_old)
        x, objective = qp_solver.solve_with_new_constraint(vertex, a_new, b_new)
    previous_solutions_to_reuse[vertex.tobytes()] = {'solver': qp_solver, 'H': (A_all, b_all), 'x': x, 'objective': objective}
    return x, objective, previous_solutions_to_reuse

