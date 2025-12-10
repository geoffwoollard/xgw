import numpy as np
from numpy.linalg import qr
import ot

def iteration_loop(mu, nu, P_plus, P_minus, e_base, R):
    x_0, v_0, objective = Hausdorff(P_plus, P_minus)
    g = new_direction(x_0, v_0, P_minus)
    g_hat, g_star = compute_hyperplane(mu, nu, g, e_base, R)
    P_plus, P_minus = update_box(P_plus, P_minus, [g, g_hat], [g_star])
    return P_plus, P_minus, objective


def run_approx(mu, nu, space_x, space_y, niter=100, epsilon=0.1):
    e_base, R = construct_basis_eij(space_x, space_y)
    P_plus, P_minus = initial_box(e_base, mu, nu)
    for iter in range(niter):
        P_plus, P_minus, objective = iteration_loop(mu, nu, P_plus, P_minus, e_base, R)
        if objective < epsilon:
            break
    return P_plus, P_minus, objective
        

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

def test_construct_basis(n1, n2):
    lin = np.linspace(0, 2, n1)
    xx, yy = np.meshgrid(lin, lin)
    space_x = np.vstack([xx.ravel(), yy.ravel()]).T
    lin = np.linspace(-1, 1, n2)
    xx, yy = np.meshgrid(lin, lin)
    space_y = np.vstack([xx.ravel(), yy.ravel()]).T
    
    N, dx = space_x.shape
    M, dy = space_y.shape
    f_base = np.zeros((N, M, dx*dy)) # constructing the function base f_{i,j}
    for i in range(dx):
        for j in range(dy):
            f_base[:,:, i+j*dx] = np.outer(space_x[:,i], space_y[:,j])
    
    e_base, R = construct_basis_eij(space_x, space_y)
    
    return np.all(np.isclose(e_base@R, f_base))

class DoubleRepresentation():
    def __init__(self):
        self.V = []
        self.H = []
        pass

    def get_H(self):
        return self.H

    def get_V(self):
        return self.V

    def H_to_V():
        '''does the update'''
        pass

    def V_to_H():
        '''does the update'''
        pass
        
    def __len__(self):
        # number of vertices and constraints
        return len(self.V), len(self.H) 
    
    def add_V(self, vertex):
        self.V.append(vertex)
    
    def add_H(self, vector, scalar):
        self.H.append([vector,scalar])
    
    def get_centroid(self):
        return np.mean(np.array(self.V))
    
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

def projection(pi, e_base):
    return np.einsum('ijk,ij->k', e_base, pi)


def function_to_cost(g, e_base):
    return np.einsum('ijk,k->ij', e_base, g)

def update_box(P_plus, P_minus, half_plans_list, vertex_list):
    P_plus.add_H(half_plans_list)
    P_minus.add_V(vertex_list)
    P_minus.V_to_H()
    P_plus.H_to_V()
    return P_plus, P_minus

def new_direction(x_0, v_0, P_minus):
    # return v_0 - P_minus.get_centroid()
    return v_0-x_0

#we can  probably  use numba here, else it may be slow, not sure how numba works with classes though
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
    # run_approx()
    print(test_construct_basis(2, 3))