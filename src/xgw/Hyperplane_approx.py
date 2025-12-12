import numpy as np
from numpy.linalg import qr
import ot
import logging
logger = logging.getLogger(__name__)

try:
    from pypoman import compute_polytope_halfspaces, compute_polytope_vertices
except ImportError as e:
    logger.info("pypoman is required for Hyperplane_approx module. Please install it via pip: pip install pypoman")

def iteration_loop(mu, nu, P_plus, P_minus, e_base, previous_solutions_to_reuse, emd_kwargs):
    x_0, v_0, objective, previous_solutions_to_reuse = Hausdorff(P_plus, P_minus, previous_solutions_to_reuse)
    g = new_direction(x_0, v_0, P_minus)
    g_hat, g_star = compute_hyperplane(mu, nu, g, e_base, emd_kwargs)
    P_plus, P_minus = update_box(P_plus, P_minus, [[g, g_hat]], [g_star])
    return P_plus, P_minus, objective, previous_solutions_to_reuse, x_0, v_0


def run_approx(mu, nu, space_x, space_y, emd_kwargs, niter=100, epsilon=1e-15):
    e_base, R = construct_basis_eij(space_x, space_y)
    P_plus, P_minus = initial_box(e_base, mu, nu, emd_kwargs)
    print('box initialized')
    
    objective_list = []
    x_0_list, v_0_list = [], []
    for iter in range(niter):
        previous_solutions_to_reuse = {} # todo: fix bug with reusing previous solutions
        print(iter)
        P_plus, P_minus, objective, _, x_0, v_0 = iteration_loop(mu, nu, P_plus, P_minus, e_base, previous_solutions_to_reuse, emd_kwargs)
        objective_list.append(objective)
        x_0_list.append(x_0)
        v_0_list.append(v_0)
        logger.info(f'Iteration {iter}, Hausdorff distance: {objective}')
        if objective < epsilon:
            break
    return P_plus, P_minus, objective, previous_solutions_to_reuse, objective_list, x_0_list, v_0_list
        

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


class DoubleRepresentation():
    def __init__(self):
        self.V = []
        self.H = ()
        self.duplicate_tol = 1e-8

    def remove_duplicates_V(self):
        """
        Remove duplicates from self.V up to a Euclidean distance tolerance.
        """
        V_array = np.array(self.V)
        keep = []
        
        for i, v in enumerate(V_array):
            if not any(np.linalg.norm(v - np.array(V_array[j])) < self.duplicate_tol for j in keep):
                keep.append(i)
        
        self.V = [V_array[i] for i in keep]

    def H_to_V(self):
        A, b = self.H
        self.V = compute_polytope_vertices(A, b)
        # self.remove_duplicates_V()

    def V_to_H(self):
        A, b = compute_polytope_halfspaces(self.V)
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
        return np.mean(np.array(self.V))
    
def initial_box(e_base, mu, nu, emd_kwargs):
    '''
    Docstring for initial_box
    
    :param e: orthonormal basis for V
    :param mu: marginal 1
    :param nu: marginal 2
    creates an initial rectangle bounding 
    '''
    P_plus, P_minus = DoubleRepresentation(), DoubleRepresentation()
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
    update_box(P_plus, P_minus, half_plans_list, vertex_list)
            
    return P_plus, P_minus

def compute_hyperplane(mu, nu, g, e_base, emd_kwargs):
    cost_matrix = function_to_cost(g, e_base)
    map, log = ot.emd(mu, nu, M=-cost_matrix, log=True, **emd_kwargs)
    return -log['cost'], projection(map, e_base)


def projection(pi, e_base):
    return np.einsum('ijk,ij->k', e_base, pi).reshape(-1,)


def function_to_cost(g, e_base):
    return np.einsum('ijk,k->ij', e_base, g)


def update_box(P_plus, P_minus, half_plans_list, vertex_list):
    P_plus.add_H(half_plans_list)
    P_minus.add_V(vertex_list)
    P_minus.V_to_H()
    P_plus.H_to_V()
    return P_plus, P_minus


def new_direction(x_0, v_0, P_minus):
    if np.allclose(x_0, v_0):
        sol = v_0 - P_minus.get_centroid()
        assert np.all(sol != 0), f' zero direction'
        return v_0 - P_minus.get_centroid()
    else:
        sol = v_0-x_0
    return (sol)/np.linalg.norm(sol)


#we can  probably  use numba here, else it may be slow, not sure how numba works with classes though
def Hausdorff(P_plus, P_minus, previous_solutions_to_reuse):
    cost = -np.inf
    for vertex in P_plus.V:
        x, objective, _ = solve_dist(vertex, P_minus, previous_solutions_to_reuse)
        # x, objective = solve_dist_brute_force(vertex, P_minus)
        if objective > cost:
            x0, v_0 = x, vertex
            cost = objective
    print(f'Hausdorff distance :{objective}')
    return x0, v_0, objective, previous_solutions_to_reuse


# def solve_dist_brute_force(vertex, P_minus, tol = 1e-8):
#     A, b = P_minus.H
#     V_list = P_minus.V
#     opt_x = None
#     objective = + np.inf
#     for i, elem  in enumerate(A):
#         dist = np.dot(elem, vertex)- b[i]
#         proj = vertex - (dist)*elem 
#         if np.all(A@proj<=b+tol) and dist<objective :
#             opt_x = proj
#             objective = dist
#     for V in V_list:
#         dist = np.linalg.norm(vertex-V)
#         if  dist<objective:
#             opt_x = V
#             objective = dist
    
#     return opt_x,objective
            
            
def P_plus_outside_P_minus(P_plus, P_minus):
    '''Check P_minus is included in P_plus'''
    A,b = P_plus.H
    residuals = [A@elem - b for elem in P_minus.V]
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

def solve_dist(vertex, P_minus, previous_solutions_to_reuse):
    from .qp_incremental_projector import IncrementalQPProjector
    if vertex.tobytes() not in previous_solutions_to_reuse:
        A_all, b_all = P_minus.H
        qp_solver = IncrementalQPProjector(dim=len(vertex), A=A_all, b=b_all)
        x, objective = qp_solver.solve(vertex)
    else:
        qp_solver = previous_solutions_to_reuse[vertex.tobytes()]['solver']
        A_old, b_old = previous_solutions_to_reuse[vertex.tobytes()]['H']
        A_all, b_all = P_minus.H
        a_new, b_new = build_new_constraint(A_all, b_all, A_old, b_old)
        x, objective = qp_solver.solve_with_new_constraint(vertex, a_new, b_new)
    previous_solutions_to_reuse[vertex.tobytes()] = {'solver': qp_solver, 'H': (A_all, b_all), 'x': x, 'objective': objective}
    return x, objective, previous_solutions_to_reuse

def minimal_test_2d():
    pass

