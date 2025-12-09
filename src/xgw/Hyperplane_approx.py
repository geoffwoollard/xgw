import numpy as np
from scipy.linalg import qr, inv
import ot

def iteration_loop(mu, nu, P_plus, P_minus, e, U):
    x_0, v_0, objective = Hausdorff(P_plus, P_minus)
    g = new_direction(x_0, v_0, P_minus)
    g_hat, g_star = compute_hyperplane(mu, nu, g, e, U)
    P_plus, P_minus = update_box(P_plus, P_minus, [g, g_hat], [g_star])
    return P_plus, P_minus, objective


def run_approx(mu, nu, space_x, space_y, niter=100, epsilon=0.1):
    e, U = construct_basis_eij(space_x, space_y)
    P_plus, P_minus = initial_box(e, mu, nu)
    for iter in range(niter):
        P_plus, P_minus, objective = iteration_loop(mu, nu, P_plus, P_minus, e, U)
        if objective < epsilon:
            break
    return P_plus, P_minus, objective
        

def construct_basis_eij(space_x, space_y):
    N, dx = space_x.shape
    M, dy = space_y.shape
    stacked_base = np.zeros((N*M, dx*dy))
    assert dx == dy
    for i in range(dx):
        for j in range(dy):
            stacked_base[:, i+j*dx] = np.ravel(np.outer(space_x[:,i], space_y[:,j])) # flattening vectors to use QR decomposition
    Q, low_dim_cost_mat = qr(stacked_base) # finding orthonormal basis : Q=[e_1,...,e_dx*dy] orthonormal base (ei flatten), [f1,...,f_dx*dy] = Q@low_dim_cost_mat, low_dim_cost_mat triangular superior matrix.
    assert np.all(np.diag(low_dim_cost_mat)!=0), f"at least one marginal is supported on a d-1 vector space" 
    e_ijs_in_PI = np.reshape(Q,(N, M, dx*dy)) # check if not  the reshape does not break order here (looks ok from 1 small test)
    return e_ijs_in_PI, np.transpose(low_dim_cost_mat)   


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
    
def initial_box(e, mu, nu, U):
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
    
    
    for e_i in e:
        for sigma in [-1,1]:
            g_hat, g_star = compute_hyperplane(mu, nu, sigma*e_i, e, U)
            vertex_list.append(g_star)
            half_plans_list.append([sigma*e, g_hat])
    update_box(P_plus, P_minus, half_plans_list, vertex_list)
            
    return  P_plus, P_minus

def compute_hyperplane(mu, nu, g, e, U):
    cost_matrix = function_to_cost(g, e, U)
    cost, log = ot.emd2(mu, nu, M=cost_matrix, log=True)
    return cost, projection(log['T'])

def projection(pi, e):
    # einsum..
    pass

def function_to_cost(g, e, U):
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

def Hausdorff(P_plus, P_minus):
    cost = np.inf
    for vertex in P_plus.get_V():
        x, v, objective = solve_dist(vertex, P_minus)
        if objective < cost:
            x0, v_0 = x, v
            cost = objective
    return x0, v_0, objective

def solve_dist(vertex, P_minus):
    pass # active sets QP HAMMER

def minimal_test_2d():
    pass

if __name__ == '__main__':
    run_approx()