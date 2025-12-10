import numpy as np
from scipy.linalg import rq
import ot

def iteration_loop(mu, nu, P_plus, P_minus, e_base, R, previous_solutions_to_reuse):
    x_0, v_0, objective, previous_solutions_to_reuse = Hausdorff(P_plus, P_minus, previous_solutions_to_reuse)
    g = new_direction(x_0, v_0, P_minus)
    g_hat, g_star = compute_hyperplane(mu, nu, g, e_base, R)
    P_plus, P_minus = update_box(P_plus, P_minus, [g, g_hat], [g_star])
    return P_plus, P_minus, objective, previous_solutions_to_reuse


def run_approx(mu, nu, space_x, space_y, niter=100, epsilon=0.1):
    e_base, R = construct_basis_eij(space_x, space_y)
    P_plus, P_minus = initial_box(e_base, mu, nu)
    previous_solutions_to_reuse = None
    for iter in range(niter):
        P_plus, P_minus, objective, previous_solutions_to_reuse = iteration_loop(mu, nu, P_plus, P_minus, e_base, R, previous_solutions_to_reuse)
        if objective < epsilon:
            break
    return P_plus, P_minus, objective, previous_solutions_to_reuse
        

def construct_basis_eij(space_x, space_y):
    N, dx = space_x.shape
    M, dy = space_y.shape
    stacked_base = np.zeros((N*M, dx*dy))
    assert dx == dy
    for i in range(dx):
        for j in range(dy):
            stacked_base[:, i+j*dx] = np.ravel(np.outer(space_x[:,i], space_y[:,j])) # flattening vectors to use QR decomposition, computing f_{i,j}
    R, Q = rq(stacked_base) # finding orthonormal basis : Q=[e_1,...,e_dx*dy] orthonormal base (ei flatten), [f1,...,f_dx*dy] = R@Q, R triangular superior matrix.
    assert np.all(np.diag(R)!=0), f"at least one marginal is supported on a d-1 vector space" 
    e_base = np.reshape(Q,(N, M, dx*dy)) 
    return e_base, np.transpose(R)


class DoubleRepresentation():
    def __init__():
        pass

    def get_H():
        pass

    def get_V():
        pass

    def H_to_V():
        '''does the update'''
        pass

    def V_to_H():
        '''does the update'''
        pass
        
    def __len__():
        pass
    
    def add_V(vertex):
        pass
    
    def add_H(vector, scalar):
        pass
    
def initial_box(e_base, mu, nu, R):
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
    
    
    for e_i in e_base:
        for sigma in [-1,1]:
            g_hat, g_star = compute_hyperplane(mu, nu, sigma*e_i, e_base, R)
            vertex_list.append(g_star)
            half_plans_list.append([sigma*e_i, g_hat])
    update_box(P_plus, P_minus, half_plans_list, vertex_list)
            
    return  P_plus, P_minus

def compute_hyperplane(mu, nu, g, e_base, R):
    cost_matrix = function_to_cost(g, e_base, R)
    cost, log = ot.emd2(mu, nu, M=cost_matrix, log=True)
    return cost, projection(log['T'])

def projection(pi, e):
    # einsum..
    pass

def function_to_cost(g, e_base, R):
    # einsum..
    pass

def update_box(P_plus, P_minus, half_plans_list, vertex_list):
    P_plus.add_H(half_plans_list)
    P_minus.add_V(vertex_list)
    P_minus.V_to_H()
    P_plus.H_to_V()
    return P_plus, P_minus

def new_direction(x_0, v_0, P_minus):
    return v_0-x_0

def Hausdorff(P_plus, P_minus, previous_solutions_to_reuse):
    cost = np.inf
    for vertex in P_plus.get_V():
        x, objective, _ = solve_dist(vertex, P_minus, previous_solutions_to_reuse)
        if objective < cost:
            x0, v_0 = x, vertex
            cost = objective
    return x0, v_0, objective, previous_solutions_to_reuse

def build_constraints(P_minus):
    pass

def build_new_constraint(P_minus):
    pass

def solve_dist(vertex, P_minus, previous_solutions_to_reuse):
    from .qp_incremental_projector import IncrementalQPProjector
    if previous_solutions_to_reuse is None:
        A, b = build_constraints(P_minus)
        qp_solver = IncrementalQPProjector(dim=len(vertex), A=A, b=b)
        x, objective = qp_solver.solve(vertex)
        previous_solutions_to_reuse = {vertex.tobytes(): qp_solver}
    else:
        qp_solver = previous_solutions_to_reuse[vertex.tobytes()]
        a_new, b_new = build_new_constraint(P_minus)
        x, objective = qp_solver.solve_with_new_constraint(vertex, a_new, b_new)
        previous_solutions_to_reuse[vertex.tobytes()] = qp_solver
    return x, objective, previous_solutions_to_reuse

def minimal_test_2d():
    pass

if __name__ == '__main__':
    run_approx()