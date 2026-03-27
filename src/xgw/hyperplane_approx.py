import numpy as np
from numpy.linalg import qr
from numba import njit
import ot
import logging
from itertools import product


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from pypoman import compute_polytope_halfspaces, compute_polytope_vertices
except ImportError as e:
    logger.info("pypoman is required for Hyperplane_approx module. Please install it via pip: pip install pypoman")

from .h_to_v_edges import update_edges_with_new_halfplane
from .h_to_v_popcount import ExtremePointPolytope, ExtremePointPolytopeSparse, masks_from_B


def build_B_from_H_and_V(A, b, V, tol=1e-5):
    # A shape (m, d), b shape (m,), V shape (n, d)
    B_bool = np.abs(A @ V.T - b[:, np.newaxis]) < tol
    B = B_bool.astype(int)
    return B

def unit_cube(dim):
    # Vertices
    V = np.array(list(product([0,1], repeat=dim)), dtype=float)

    # Halfspace form
    A = np.vstack([np.eye(dim), -np.eye(dim)])
    b = np.concatenate([np.ones(dim), np.zeros(dim)])
    tol = 1e-8
    B_bool = build_B_from_H_and_V(A, b, V, tol=tol)
    masks = masks_from_B(B_bool)

    return V, A, b, B_bool, masks

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
    except Exception as e:
        H = np.hstack([A, b.reshape(-1, 1)]) # cdd format
        for decimals in range(decimals_start, decimals_end, -1):
            V = []
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
                except RuntimeError as _:
                    # 4. If it fails due to numerical issues, add tiny jitter
                    scale = 0.5 * 10**(-decimals)
                    jitter = scale * np.random.randn(*H_rounded_unique.shape)
                    H_perturbed = H_rounded_unique + jitter
                    A_perturbed = H_perturbed[:,:-1]
                    b_perturbed = H_perturbed[:,-1]
                    V = compute_polytope_vertices(A_perturbed, b_perturbed)

            except Exception as _:
                # print(f'Failed with rounding to {decimals} decimals: {e}')
                continue  # try next lower precision

            else:
                if len(V)>1:
                    # print(f"Success with {decimals} decimals!")
                    break
        raise RuntimeError(f"Failed to compute vertices from halfspaces with stable rounding, even with jitter. Last error: {e}")
    return V

def stabilize_compute_polytope_halfspaces(V, decimals_start=18, decimals_end=3):
    '''Try to compute halfspaces from vertices with decreasing rounding precision to enhance numerical stability.
    
    This function avoids the error Error: Numerical inconsistency is found.  Use the GMP exact arithmetic.
    This error is related to how cdd casts floating point numbers to rationals internally, which can lead to numerical issues.
    In general, rounding more allows the function to succeed, but too much rounding can distort the polytope shape.
    Also, aggresive rounding can fail, while moderate rounding with tiny jitter can succeed.
    Hence the strategy of starting from high precision and decreasing it, trying jitter if needed.
    The jitter is scaled to be small compared to the rounding scale (rounding scale is 1/2 std of the jitter, per component).
    
    '''
    try:
        A, b = compute_polytope_halfspaces(V) # TODO: figure out why fails for circle
    except:
        for decimals in range(decimals_start, decimals_end, -1):
            A = []
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
                if len(A) > 1:
                    # print(f"Success with {decimals} decimals!")
                    break
            
    return A, b



def _run_approx(mu, nu, space_x, space_y, emd_kwargs, niter=100, epsilon=1e-15):
    e_base, R, p_plus, p_minus = initial_box(space_x, space_y, mu, nu, emd_kwargs)
    logger.info('box initialized')
    
    objective_list = []
    x_0_list = []
    v_0_list = []
    for iter in range(niter):
        logger.info(f'Iteration {iter}')
        p_plus, p_minus, objective, x_0, v_0 = iteration_loop_geometric(mu, nu, p_plus, p_minus, e_base, emd_kwargs, ret_arg=True)
        objective_list.append(objective)
        x_0_list.append(x_0)
        v_0_list.append(v_0)
        logger.info(f'Iteration {iter}, hausdorff distance: {objective}')
        if objective < epsilon:
            break
    return p_plus, p_minus, objective, objective_list, x_0_list, v_0_list

def _run_approx_yield(mu, nu, space_x, space_y, emd_kwargs, niter=100, epsilon=1e-15):
    e_base, R, p_plus, p_minus = initial_box(space_x, space_y, mu, nu, emd_kwargs)
    logger.info('box initialized')
    
    for iter in range(niter):
        p_plus, p_minus, objective, _, _ = iteration_loop_geometric(mu, nu, p_plus, p_minus, e_base, emd_kwargs, ret_arg=True)
        yield(p_plus, p_minus, objective)
        if objective < epsilon:
            break

# def iteration_loop_hausdorff(mu, nu, p_plus, p_minus, e_base, previous_solutions_to_reuse, emd_kwargs):
#     #deprecated
#     x_0, v_0, objective, previous_solutions_to_reuse = hausdorff(p_plus, p_minus, previous_solutions_to_reuse)
#     g = new_direction(x_0, v_0, p_minus)
#     g_hat, g_star = compute_hyperplane(mu, nu, g, e_base, emd_kwargs)
#     p_plus, p_minus = update_box(p_plus, p_minus, [[g, g_hat]], [g_star])
#     return p_plus, p_minus, objective, previous_solutions_to_reuse

def iteration_loop_geometric(mu, nu, P_plus, P_minus, e_base, emd_kwargs, ret_arg=False):
    tup = new_direction(P_plus, P_minus, ret_arg=ret_arg)
    g, objective = tup[0], tup[1]
    g_hat, g_star = compute_hyperplane(mu, nu, g, e_base, emd_kwargs)
    P_plus, P_minus = update_box(P_plus, P_minus, [[g, g_hat]], [g_star])
    if ret_arg:
        return P_plus, P_minus, objective, tup[2], g_star
    return P_plus, P_minus, objective


def run_approx(mu, nu, space_x, space_y, emd_kwargs, niter=100, epsilon=1e-15):
    e_base, R, p_plus, p_minus = initial_box(space_x, space_y, mu, nu, emd_kwargs)
    logger.info('box initialized')
    
    objective_list = []
    for iter in range(niter):
        logger.info(f'Iteration {iter}')
        p_plus, p_minus, objective = iteration_loop_geometric(mu, nu, p_plus, p_minus, e_base, emd_kwargs)
        objective_list.append(objective)
        logger.info(f'Iteration {iter}, hausdorff distance: {objective}')
        if objective < epsilon:
            break
    return p_minus, p_plus, objective, R, e_base


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

def classical_gw_construct_basis_eij(space_x, space_y, g_func):
    # finding orthonormal basis : Q = [e_1,...,e_dx*dy] orthonormal base (ei flatten), f_base = [f1,...,f_dx*dy] = e_base@R, R triangular superior matrix of size dx*dy^2, e_base (N, M) by dx*dy tensor, f_base (N, M) by dx*dy tensor
    N, dx = space_x.shape
    M, dy = space_y.shape
    stacked_base = np.zeros((N*M, dx*dy+1))
    for i in range(dx):
        for j in range(dy):
            stacked_base[:, i+j*dx] = np.ravel(np.outer(space_x[:,i], space_y[:,j])) # flattening vectors to use QR decomposition
    stacked_base[:, -1] = np.ravel(g_func)
    Q, R = qr(stacked_base, mode='reduced') 
    assert np.all(np.diag(R)!=0), f"at least one marginal is supported on a d-1 vector space" 
    e_base = np.reshape(Q,(N, M, dx*dy+1)) # reshape to  physical dimensions, now we have a N by M by dx*dy+1 tensor
    return e_base, R

@njit
def e_to_f(vect, R, d):
    sol = (vect@R).reshape((d,d))
    return sol


@njit
def f_to_e(vect, R_inv):
    return np.ravel(vect) @ R_inv


class DoubleDescription():
    def __init__(self, 
                 duplicate_tol=1e-5, 
                 implementation='cdd', 
                 E_initialization=None, 
                 V_initialization=None, 
                 B_initialization=None, 
                 masks_initialization=None, 
                 A_initialization=None, 
                 b_initialization=None, 
                 dual_implementation=None):
        self.V = []
        self.H = ()
        self.duplicate_tol = duplicate_tol
        self.implementation = implementation
        self.dual_implementation = dual_implementation
        if self.implementation == 'cdd':
            pass
        elif self.implementation == 'h_to_v_edges':
            self.E = E_initialization
        elif self.implementation == 'h_to_v_popcount':
            self.poly = ExtremePointPolytope(E=V_initialization, B=B_initialization, dim=V_initialization.shape[1])
        elif self.implementation == 'h_to_v_popcount_sparse':
            self.poly = ExtremePointPolytopeSparse(E=V_initialization, masks=masks_initialization, A=A_initialization, b=b_initialization, )
        elif self.implementation == 'v_to_h_dual':
            assert self.dual_implementation is not None, 'dual_implementation is required for v_to_h_dual implementation'
            assert self.dual_implementation in ['cdd', 'h_to_v_popcount'], f'Unknown dual implementation {self.dual_implementation} for v_to_h_dual implementation'
            from xgw.v_to_h_dual import setup_dual_polytope
            dd_dual_polytope, center, dim = setup_dual_polytope(V_initialization, A_initialization, b_initialization, implementation=self.dual_implementation)
            self.dim = dim
            self.dd_dual_polytope = dd_dual_polytope
            self.center = center

        else:
            raise NotImplementedError(f'{self.implementation} implementation is not implemented yet')
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

        if self.implementation == 'h_to_v_edges':
            # decrement edges for removed vertices
            index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}
            self.E = np.array([(index_map[edge[0]], index_map[edge[1]]) for edge in self.E if edge[0] in index_map and edge[1] in index_map])
            logger.info(f"Re-indexed edges, new number of edges: {len(self.E)}")

        elif self.implementation == 'h_to_v_popcount':
            self.poly.E = self.poly.E[unique_indices]
            self.poly.B = self.poly.B[:, unique_indices]
            self.poly.rebuild_adjacency() # TODO: remove if test blow is passing
            assert np.allclose(self.poly.D, self.poly.D[np.ix_(unique_indices, unique_indices)])
            logger.info(f"Re-indexed active-constraint matrix, new shape: {self.poly.B.shape}")
        
        elif self.implementation == 'h_to_v_popcount_sparse':
            # TODO: check correctness of this, especially the re-indexing of masks and adjacency
            self.poly.E = self.poly.E[unique_indices]
            self.poly.masks = self.poly.masks[unique_indices]
            self.poly.D = self.poly._build_adjacency()
            assert np.allclose(self.poly.D, self.poly.D[np.ix_(unique_indices, unique_indices)])
            logger.info(f"Re-indexed masks, new shape: {self.poly.masks.shape}")

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
            raise NotImplementedError(f'{self.implementation} implementation is not implemented yet')
        elif self.implementation == 'h_to_v_popcount':
            raise NotImplementedError(f'{self.implementation} implementation is not implemented yet')
        elif self.implementation == 'h_to_v_popcount_sparse':
            raise NotImplementedError(f'{self.implementation} implementation is not implemented yet')
        elif self.implementation == 'cdd':
            A, b = self.H
            logger.info(f'Computing vertices from half-planes: A shape {A.shape}, b shape {b.shape}')
            # logger.info(f'Half-planes: {A}, {b}')
            assert not self.check_feasibility_V(A, b)
            # logger.info(f'A,b: {A, b}')
            self.V = stabilize_compute_polytope_vertices(A, b)
            self.remove_duplicates_V()

    def V_to_H(self):
        assert self.implementation == 'cdd', f'{self.implementation} implementation only supports V_to_H via cdd'
        # loop over high to low decimals to ensure numerical stability. take largest that works
        logger.info(f'V_to_H Computing half-planes from vertices: number of vertices {len(self.V)}')
        A, b = stabilize_compute_polytope_halfspaces(self.V)
        logger.info(f'V_to_H Computed half-planes: A shape {A.shape}, b shape {b.shape}')
        a_norm = np.linalg.norm(A, axis=1)
        b /= a_norm
        A /= a_norm[:, np.newaxis] # TODO: fix: RuntimeWarning: divide by zero encountered in divide
        self.H = (A,b)
        
    def __len__(self):
        # number of vertices and constraints
        return len(self.V), self.H[0].shape 
    
    def __str__(self):
        return f"Vertices: {self.V}\nHalf-planes: {self.H}"
    
    # could be optimized for a family of vertices
    def add_V(self, vertex_list):
        if self.implementation == 'v_to_h_dual':
            assert len(vertex_list) == 1, f'{self.implementation} implementation only supports adding one vertex at a time'
            x_new = vertex_list[0]
            from xgw.v_to_h_dual import add_vertex_via_dual
            vertices, A, b, dd_dual_polytope = add_vertex_via_dual(x_new, self.center, self.dd_dual_polytope, self.dim)
            self.dd_dual_polytope = dd_dual_polytope
            self.V = vertices.tolist()
            self.H = [A, b]

        elif self.implementation == 'h_to_v_edges':
            raise NotImplementedError(f'{self.implementation} implementation is not implemented yet')
        elif self.implementation == 'h_to_v_popcount':
            raise NotImplementedError(f'{self.implementation} implementation is not implemented yet')
        elif self.implementation == 'h_to_v_popcount_sparse':
            raise NotImplementedError(f'{self.implementation} implementation is not implemented yet')
        elif self.implementation == 'cdd':
            # vertex is a d^2 by 1 vector
            self.V.extend(vertex_list)
            self.remove_duplicates_V()
            self.V_to_H()
    
    # could be optimized for a family of vectors and scalars
    def add_H(self, constraint_list):
        if self.implementation == 'h_to_v_popcount_sparse': # TODO: refactor to avoid code duplication with h_to_v_popcount, since the only difference is the type of self.poly
            assert len(constraint_list) == 1, f'{self.implementation} implementation only supports adding one half-plane at a time'
            a_new, b_new = constraint_list[0]
            self.poly.add_constraint(a_new, b_new)
            self.V = [np.array(v) for v in self.poly.E]
            A, b = self.H
            A_all = np.vstack([A, a_new.reshape(-1,)])
            b_all = np.hstack([b, b_new])
            self.H = [A_all, b_all]
        elif self.implementation == 'h_to_v_popcount':
            assert len(constraint_list) == 1, f'{self.implementation} implementation only supports adding one half-plane at a time'
            a_new, b_new = constraint_list[0]
            self.poly.add_constraint(a_new, b_new)
            self.V = [np.array(v) for v in self.poly.E]
            A, b = self.H
            A_all = np.vstack([A, a_new.reshape(-1,)])
            b_all = np.hstack([b, b_new])
            self.H = [A_all, b_all]
        elif self.implementation == 'h_to_v_edges':
            assert len(constraint_list) == 1, f'{self.implementation} implementation only supports adding one half-plane at a time'
            a_new, b_new = constraint_list[0]
            V = np.array(self.V)
            A, b = self.H
            E = self.E
            V_final, E_final, A_final, b_final = update_edges_with_new_halfplane(V, E, A, b, a_new, b_new)
            self.V = V_final.tolist()
            self.E = E_final
            self.H = [A_final, b_final]
            self.remove_duplicates_V() 
        elif self.implementation == 'cdd':
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


def initial_box(space_x, space_y, mu, nu, emd_kwargs, p_plus_implementation='cdd', p_minus_implementation='v_to_h_dual', p_minus_dual_implementation='h_to_v_popcount'):
    '''
    Docstring for initial_box
    
    :param e: orthonormal basis for V
    :param mu: marginal 1
    :param nu: marginal 2
    creates an initial rectangle bounding 
    '''
    e_base, R = construct_basis_eij(space_x, space_y)
    p_plus_initial = DoubleDescription(implementation='cdd')
    p_minus_initial = DoubleDescription(implementation='cdd')

    vertex_list = []
    half_plans_list = []
    c = e_base.shape[2]
    direction_list = []
    for i in range(c):
        dir = np.zeros(c)
        dir[i] = 1
        direction_list.append(dir)
    direction_list.append(-np.ones(c))
    for dir in direction_list:
        g_hat, g_star = compute_hyperplane(mu, nu, dir, e_base, emd_kwargs)
        vertex_list.append(g_star)
        half_plans_list.append([dir, g_hat])
            
            
    update_box(p_plus_initial, p_minus_initial, half_plans_list, vertex_list)
    if p_minus_implementation == 'cdd':
        p_minus = p_minus_initial
    elif p_minus_implementation == 'v_to_h_dual':
        A_p_minus, b_p_minus = p_minus_initial.H
        p_minus = DoubleDescription(implementation=p_minus_implementation, 
                                                dual_implementation=p_minus_dual_implementation, 
                                                V_initialization=np.array(p_minus_initial.V),
                                                A_initialization=A_p_minus, 
                                                b_initialization=b_p_minus)
        p_minus.H = p_minus_initial.H
        p_minus.V = p_minus_initial.V
    else:
        raise NotImplementedError(f'{p_minus_implementation} implementation is not implemented yet')

    if p_plus_implementation == 'h_to_v_edges':
        from xgw.h_to_v_edges import find_edges
        E_initialization = find_edges(p_plus_initial.V) if p_plus_implementation == 'h_to_v_edges' else None
        p_plus = DoubleDescription(implementation=p_plus_implementation, E_initialization=E_initialization)
        p_plus.V = p_plus_initial.V
        p_plus.H = p_plus_initial.H
    elif p_plus_implementation == 'cdd':
        p_plus = p_plus_initial
    elif p_plus_implementation == 'h_to_v_popcount':
        A, b = p_plus_initial.H
        V_initialization = np.array(p_plus_initial.V)
        tol = 1e-5
        B_bool = np.abs(A @ V_initialization.T - b[:, np.newaxis]) < tol
        B_initialization = B_bool.astype(int)
        p_plus = DoubleDescription(implementation=p_plus_implementation, 
                                   V_initialization=V_initialization, 
                                   B_initialization=B_initialization
                                   )
        p_plus.V = p_plus_initial.V
        p_plus.H = p_plus_initial.H
    elif p_plus_implementation == 'h_to_v_popcount_sparse':
        A, b = p_plus_initial.H
        V_initialization = np.array(p_plus_initial.V)
        tol = 1e-5
        B_bool = np.abs(A @ V_initialization.T - b[:, np.newaxis]) < tol

        masks_initialization = masks_from_B(B_bool)
        p_plus = DoubleDescription(implementation=p_plus_implementation,
                                   V_initialization=V_initialization,
                                   masks_initialization=masks_initialization,
                                   A_initialization=A,
                                   b_initialization=b)
        p_plus.V = [np.array(v) for v in p_plus.poly.E]
        p_plus.H = [A, b]
            
    return e_base, R, p_plus, p_minus

def classical_gw_initial_box(space_x, space_y, g_func, mu, nu, emd_kwargs, p_plus_implementation='cdd'):
    '''
    Docstring for initial_box
    
    :param e: orthonormal basis for V
    :param mu: marginal 1
    :param nu: marginal 2
    creates an initial rectangle bounding 
    '''
    e_base, R = classical_gw_construct_basis_eij(space_x, space_y, g_func)
    p_plus_initial, p_minus = DoubleDescription(implementation='cdd'), DoubleDescription(implementation='cdd')
    vertex_list = []
    half_plans_list = []
    c = e_base.shape[2]
    for i in range(c):
        for sigma in [-1,1]:
            e_i = np.zeros(c)
            e_i[i] = 1
            g_hat, g_star = compute_hyperplane(mu, nu, sigma*e_i, e_base, emd_kwargs)
            vertex_list.append(g_star)
            half_plans_list.append([sigma*e_i, g_hat])
    update_box(p_plus_initial, p_minus, half_plans_list, vertex_list)
    if p_plus_implementation == 'h_to_v_edges':
        from xgw.h_to_v_edges import find_edges
        E_initialization = find_edges(p_plus_initial.V) if p_plus_implementation == 'h_to_v_edges' else None
        p_plus = DoubleDescription(implementation=p_plus_implementation, E_initialization=E_initialization)
        p_plus.V = p_plus_initial.V
        p_plus.H = p_plus_initial.H
    elif p_plus_implementation == 'cdd':
        p_plus = p_plus_initial
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
    if p_plus.implementation == 'cdd':
        logger.info('Adding new half-planes and vertices to the bounding boxes')
        p_plus.add_H(half_planes_list)
        logger.info('Added half-planes to p_plus')
        # p_plus.H_to_V()
        # logger.info('Updated vertices of p_plus from half-planes')
    elif p_plus.implementation == 'h_to_v_edges':
        assert len(half_planes_list) == 1, 'h_to_v_edges implementation only supports adding one half-plane at a time'
        a_new, b_new = half_planes_list[0]
        p_plus.add_H([[a_new, b_new]])
    elif p_plus.implementation == 'h_to_v_popcount':
        assert len(half_planes_list) == 1, f'{p_plus.implementation} implementation only supports adding one half-plane at a time'
        a_new, b_new = half_planes_list[0]
        p_plus.add_H([[a_new, b_new]])
    elif p_plus.implementation == 'h_to_v_popcount_sparse':
        assert len(half_planes_list) == 1, f'{p_plus.implementation} implementation only supports adding one half-plane at a time'
        a_new, b_new = half_planes_list[0]
        p_plus.add_H([[a_new, b_new]])

        
    if p_minus.implementation == 'cdd':
        p_minus.add_V(vertex_list)
        logger.info('Added vertices to p_minus')
        p_minus.V_to_H()
        logger.info('Updated half-planes of p_minus from vertices')
    elif p_minus.implementation == 'h_to_v_edges':
        raise NotImplementedError(f'{p_minus.implementation} implementation is not implemented yet')
    elif p_minus.implementation == 'h_to_v_popcount':
        raise NotImplementedError(f'{p_minus.implementation} implementation is not implemented yet')
    elif p_minus.implementation == 'h_to_v_popcount_sparse':
        raise NotImplementedError(f'{p_minus.implementation} implementation is not implemented yet')
    elif p_minus.implementation == 'v_to_h_dual':
        assert len(vertex_list) == 1, f'{p_minus.implementation} implementation only supports adding one vertex at a time'
        x_new = vertex_list[0]
        p_minus.add_V([x_new])
    else:
        raise NotImplementedError(f'{p_minus.implementation} implementation is not implemented yet')
    return p_plus, p_minus


# def new_direction_old(x_0, v_0, p_minus):
#     #deprecated
#     if np.allclose(x_0, v_0):
#         sol = v_0 - p_minus.get_centroid()
#         assert np.all(sol != 0), f' zero direction'
#         return v_0 - p_minus.get_centroid()
#     else:
#         sol = v_0-x_0
#     return (sol)/np.linalg.norm(sol)

def new_direction(P_plus, P_minus, tol=1e-10, ret_arg=False):
# Normalized normal vectors of P_minus
    A,b =  P_minus.H
    vect_list = np.array(P_plus.V)
    # Finding best direction 
    testing_dir = A @ (vect_list.T) - np.tile(b,(len(vect_list), 1)).T
    logger.info(f'shape test_dir: {testing_dir.shape}')
    indexes = np.unravel_index(np.argmax(testing_dir), testing_dir.shape)
    x_plus = vect_list [indexes[1]]
    g = A[indexes[0]]
    val = g @ x_plus - b[indexes[0]]
    # checking that the point is outside P_minus
    assert val >= -tol 
    if ret_arg:
        return g / np.linalg.norm(g), val, x_plus
    return g / np.linalg.norm(g), val


# def hausdorff(p_plus, p_minus, previous_solutions_to_reuse):
#     #deprecated
#     cost = -np.inf
#     for vertex in p_plus.V:
#         x, objective, _ = solve_dist(vertex, p_minus, previous_solutions_to_reuse)
#         # x, objective = solve_dist_brute_force(vertex, p_minus)
#         if objective > cost:
#             x0, v_0 = x, vertex
#             cost = objective
#     logger.info(f'Hausdorff distance :{objective}')
#     return x0, v_0, objective, previous_solutions_to_reuse

            
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
    '''Solve the optimization problem of finding the point in p_minus that is farthest from vertex.'''
    from .qp_incremental_projector import IncrementalQPProjector
    # print(f'solve_dist for vertex {vertex}')
    # print(f'len p_minus half-planes: {len(p_minus.H[0])}')
    if vertex.tobytes() not in previous_solutions_to_reuse:
        A_all, b_all = p_minus.H
        qp_solver = IncrementalQPProjector(dim=len(vertex), A=A_all, b=b_all)
        x, objective = qp_solver.solve(vertex)
    else:
        qp_solver = previous_solutions_to_reuse[vertex.tobytes()]['solver']
        A_old, b_old = previous_solutions_to_reuse[vertex.tobytes()]['H']
        A_all, b_all = p_minus.H
        if len(A_all) > len(A_old):
            a_new, b_new = build_new_constraint(A_all, b_all, A_old, b_old) 
            # print('reusing: len A_old', len(A_old), 'len A_all', len(A_all))
            x, objective = qp_solver.solve_with_new_constraint(vertex, a_new, b_new) 
        else:
            raise ValueError('No new constraints to add, but previous solution exists. This should not happen, check the logic of when to reuse previous solutions.')
            # A_all, b_all = p_minus.H
            # qp_solver = IncrementalQPProjector(dim=len(vertex), A=A_all, b=b_all)
            # x, objective = qp_solver.solve(vertex)
    previous_solutions_to_reuse[vertex.tobytes()] = {'solver': qp_solver, 'H': (A_all, b_all), 'x': x, 'objective': objective}
    return x, objective, previous_solutions_to_reuse

