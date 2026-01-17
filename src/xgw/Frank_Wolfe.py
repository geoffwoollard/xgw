import numpy as np
from numba import njit
import ot

from .Hyperplane_approx import e_to_f

@njit
def det_23d(mat):
    l,_=mat.shape
    if l==1:
        return mat[0,0]
    elif l==2:
        return mat[0,0]*mat[1,1]-mat[0,1]*mat[1,0]
    else:
        return mat[0,0]*(mat[1,1]*mat[2,2]-mat[2,1]*mat[1,2]) + mat[0,1]*(mat[1,2]*mat[2,0]-mat[1,0]*mat[2,2]) + mat[0,2]*(mat[1,0]*mat[2,1]-mat[2,0]*mat[1,1]) 

@njit
def sq_norm(mat):
    return np.sum(mat**2)

@njit
def matrix_cofactor_low_dim(mat):
    row_l, col_l = mat.shape
    comat = np.zeros((row_l, col_l))
    for row in range(row_l):
        for col in range(col_l):
            mask_col = np.ones(row_l, dtype=np.bool)
            mask_col[col] = False
            mask_row = np.ones(row_l, dtype=np.bool)
            mask_row[row] = False
            minor = mat[mask_row,:][:,mask_col]
            comat[row, col] = (-1)**(row+col) * det_23d(minor)
    return comat

@njit
def factorial(n):
    if n == 0:
        return 1
    elif n == 1:
        return 1
    elif n == 2:
        return 2
    elif n == 3:
        return 6
    else:
        raise ValueError('need to implement')
    
###########  Functions that need to be implemented for each cost ############
@njit
def polynomial_cost(sigma, cost, t):
    if cost == 'IGW':
        return sq_norm(sigma)
    elif cost == 'DGW':
        l=len(sigma)
        return factorial(l) *det_23d(sigma)
    elif cost == 'CGW':
        l=len(sigma)
        return t*sq_norm(sigma)+ (1-t)*factorial(l) *det_23d(sigma)
    else:
        raise ValueError('Cost not implemented')

@njit  
def const_cost(sigma_x, sigma_y,  cost, t):
    return polynomial_cost(sigma_x, cost, t) + polynomial_cost(sigma_y, cost, t)

@njit
def linearized_cost_matrix(sigma, cost, t):
    if cost == 'IGW':
        return sigma
    elif cost == 'DGW':
        l=len(sigma)
        return factorial(l-1)*matrix_cofactor_low_dim(sigma)
    elif cost == 'CGW':
        l=len(sigma)
        return t*sigma + (1-t)*factorial(l-1)*matrix_cofactor_low_dim(sigma)
    else:
        raise ValueError('Cost not implemented')
#########


def linearized_cost_function(space_x, space_y, M): 
    return (space_x @ M.T).dot(space_y.T)
   
def cross_covariance(space_x, space_y, pi):
    return np.einsum('kd,lD,kl->dD', space_x, space_y, pi)

def covariance(space_x, mu):
    return np.einsum('kd,kD,k->dD', space_x, space_x, mu)

@njit
def line_search_IGW(sigma_1, sigma_0):
    n0 = sq_norm(sigma_0)
    n1 = sq_norm(sigma_1)
    if n1>n0:
            tau = 1
            T = n1
    else :
            tau = 0
            T = n0
    return T,tau

@njit
def optimize_deg_2_polynomial (alpha, beta, gamma):
    def q(x):
        return alpha*x**2 + beta*x +gamma
    
    if alpha>=0:
        if q(1)>q(0):
                tau = 1
        else :
                tau = 0
    else:
        lbd = -beta/(2*alpha)
        tau = min(1,max(0,lbd))
    T = q(tau)
    return T, tau

@njit
def optimize_deg_3_polynomial (alpha, beta, gamma, delta):
    
    if alpha == 0:
        return optimize_deg_2_polynomial (beta, gamma, delta)
    else:
        A = True
        def q(x):
            return alpha*x**3 + beta*x**2 + gamma*x + delta
        
        DELT = 4*beta**2 - 12*alpha*gamma
        if DELT>=0:
            lbd = -(2*beta+np.sqrt(DELT))/(6*alpha)
            if 0<lbd and lbd<1 and q(lbd)>q(1) and q(lbd)>q(0):
                tau = lbd
                A = False
        if A:
            if q(1)>q(0):
                    tau = 1
            else :
                    tau = 0  
        T = q(tau) 
        return T, tau

@njit
def line_search_CGW_2d(sigma_1, sigma_0, t):
    d_1 = det_23d(sigma_1)
    d_0 = det_23d(sigma_0)
    tr = np.trace(sigma_1)*np.trace(sigma_0) - np.trace(sigma_1@sigma_0)
    n1 = sq_norm(sigma_1)
    n0 = sq_norm(sigma_0)
    alpha = d_1 + d_0 - tr
    beta = tr - 2*d_0
    gamma = d_0
    
    alpha = t*(n1+n0) + 2*(1-t)*alpha
    beta = -2*t*n0 + 2*(1-t)*beta 
    gamma = t*n0 + 2*(1-t)*gamma
    
    T, tau = optimize_deg_2_polynomial (alpha, beta, gamma)
    return T,tau

@njit
def line_search_DGW_2d(sigma_1, sigma_0):
    d_1 = det_23d(sigma_1)
    d_0 = det_23d(sigma_0)
    tr = np.trace(sigma_1)*np.trace(sigma_0) - np.trace(sigma_1@sigma_0)
    alpha = 2*(d_1 + d_0 - tr)
    beta = 2*(tr - 2*d_0)
    gamma = 2*(d_0)
    
    T, tau = optimize_deg_2_polynomial (alpha, beta, gamma)
    return T,tau

@njit
def line_search_DGW_3d(sigma_1, sigma_0):
    d_1 = det_23d(sigma_1)
    d_0 = det_23d(sigma_0)
    
    tr_1 = np.trace(sigma_1)
    tr_0 = np.trace(sigma_0)
    tr_11 = np.trace(sigma_1@sigma_1)
    tr_00 = np.trace(sigma_0@sigma_0)
    tr_10 = np.trace(sigma_1@sigma_0)
    tr_110 = np.trace(sigma_1@sigma_1@sigma_0)
    tr_001 = np.trace(sigma_0@sigma_0@sigma_1)
    
    TR_0 = 1/2*((tr_1**2-tr_11)*tr_0) + tr_110 -tr_1*tr_10
    TR_1 = 1/2*((tr_0**2-tr_00)*tr_1) + tr_001 -tr_0*tr_10
    
    alpha = 6*(d_1 - d_0 - TR_1 + TR_0)
    beta = 6*(3*d_0 + TR_1 - 2*TR_0)
    gamma = 6*(-3*d_0 + TR_0)
    delta = 6*(d_0)
    T, tau = optimize_deg_3_polynomial (alpha, beta, gamma, delta)
    return T,tau

@njit
def line_search_CGW_3d(sigma_1, sigma_0, t):
    d_1 = det_23d(sigma_1)
    d_0 = det_23d(sigma_0)
    n1 = sq_norm(sigma_1)
    n0 = sq_norm(sigma_0)
    
    tr_1 = np.trace(sigma_1)
    tr_0 = np.trace(sigma_0)
    tr_11 = np.trace(sigma_1@sigma_1)
    tr_00 = np.trace(sigma_0@sigma_0)
    tr_10 = np.trace(sigma_1@sigma_0)
    tr_110 = np.trace(sigma_1@sigma_1@sigma_0)
    tr_001 = np.trace(sigma_0@sigma_0@sigma_1)
    
    TR_0 = 1/2*((tr_1**2-tr_11)*tr_0) + tr_110 -tr_1*tr_10
    TR_1 = 1/2*((tr_0**2-tr_00)*tr_1) + tr_001 -tr_0*tr_10
    
    alpha = 6*(d_1 - d_0 - TR_1 + TR_0)
    beta = 6*(3*d_0 + TR_1 - 2*TR_0)
    gamma = 6*(-3*d_0 + TR_0)
    delta = 6*(d_0)
    
    beta = t*(n1+n0) + (1-t)*beta
    gamma = -2*t*n0 + (1-t)*gamma
    delta = t*n0 + (1-t)*delta
    
    T, tau = optimize_deg_3_polynomial (alpha, beta, gamma, delta)
    return T,tau


@njit
def line_search(sigma_1, sigma_0, cost, t):
    if cost == 'IGW':
        return line_search_IGW(sigma_1, sigma_0)
    elif  cost == 'DGW' and len(sigma_0)==2:
        return line_search_DGW_2d(sigma_1, sigma_0)
    elif  cost == 'DGW' and len(sigma_0)==3:
        return line_search_DGW_3d(sigma_1, sigma_0)
    elif cost == 'CGW' and len(sigma_0)==2:
        return line_search_CGW_2d(sigma_1, sigma_0, t)
    elif cost == 'CGW' and len(sigma_0)==3:
        return line_search_CGW_3d(sigma_1, sigma_0, t)
    else:
        raise ValueError('Cost not implemented')
    
def _Frank_Wolfe_iter(mu, space_x, nu, space_y, init_direc, R, cost='IGW', iter_max=50, t=0.5):
    for it in range(iter_max):
        # Initial direction
        if it == 0:
            # init_direc is in the e basis (and not f) and it has the wrong format
            init_direc = e_to_f(init_direc, R)
            sigma_pi_n = init_direc
        else:
            sigma_pi_n = cross_covariance(space_x, space_y, pi_n)
            
        M_pi_n = linearized_cost_matrix(sigma_pi_n, cost, t)
        lin_cost = linearized_cost_function(space_x, space_y, M_pi_n)
        pi_n_1_hat = ot.emd(mu, nu, -lin_cost)
        pi_n_1_hat, log = ot.emd(mu, nu, M=-lin_cost, log=True)
        M_pi_val = -log['cost']
        sigma_pi_n_1_hat = cross_covariance(space_x, space_y, pi_n_1_hat)
        T, tau = line_search(sigma_pi_n_1_hat, sigma_pi_n, cost, t) 
        print(f"FW it {it}: tau={tau}, {cost}^2 T cost={T}", end=' ')
        if tau == 0:
            break
        pi_n_1 = tau*pi_n_1_hat + (1-tau)*pi_n
        pi_n = pi_n_1
        yield pi_n_1_hat, M_pi_n, M_pi_val
    
    
def Frank_Wolfe_GW(mu, space_x, nu, space_y, cost='IGW', pi_n=None, iter_max = 50, t=0.5):
    if pi_n is None:
        pi_n = np.outer(mu, nu)
    
    sigma_x = covariance(space_x, mu)
    sigma_y = covariance(space_y, nu)
    initial_cost = const_cost(sigma_x, sigma_y, cost, t)
    print('initial_cost', initial_cost)
    for it in range(iter_max):
        sigma_pi_n = cross_covariance(space_x, space_y, pi_n)
        M_pi_n = linearized_cost_matrix(sigma_pi_n, cost, t)
        lin_cost = linearized_cost_function(space_x, space_y, M_pi_n)
        pi_n_1_hat = ot.emd(mu, nu, -lin_cost)
        sigma_pi_n_1_hat = cross_covariance(space_x, space_y, pi_n_1_hat)
        T, tau = line_search(sigma_pi_n_1_hat, sigma_pi_n, cost, t) 
        print(f"it {it}: tau={tau}, {cost}^2_cost={initial_cost-2*T}", end=' ')
        if tau == 0:
            break
        pi_n_1 = tau*pi_n_1_hat + (1-tau)*pi_n
        pi_n = pi_n_1
    return initial_cost-2*T, pi_n


def Frank_Wolfe_polynomial(mu, space_x, nu, space_y, pi_n, cost='IGW', iter_max = 50, t=0.5):
    for _ in range(iter_max):
        sigma_pi_n = cross_covariance(space_x, space_y, pi_n)
        M_pi_n = linearized_cost_matrix(sigma_pi_n, cost, t)
        lin_cost = linearized_cost_function(space_x, space_y, M_pi_n)
        pi_n_1_hat = ot.emd(mu, nu, -lin_cost)
        sigma_pi_n_1_hat = cross_covariance(space_x, space_y, pi_n_1_hat)
        T, tau = line_search(sigma_pi_n_1_hat, sigma_pi_n, cost, t) 
        if tau == 0:
            break
        pi_n_1 = tau*pi_n_1_hat + (1-tau)*pi_n
        pi_n = pi_n_1
    return T, pi_n
    


   
