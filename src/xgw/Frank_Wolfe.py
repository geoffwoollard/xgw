import numpy as np
from numba import njit
from math import factorial
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


###########  Functions that need to be implemented for each cost ############
def polynomial_cost(sigma, t, cost='IGW'):
    if cost == 'IGW':
        return np.linalg.norm(sigma)**2
    elif cost == 'DGW':
        l=len(sigma)
        return factorial(l) *det_23d(sigma)
    elif cost == 'IDGW':
        l=len(sigma)
        return t*np.linalg.norm(sigma)**2+ (1-t)*factorial(l) *det_23d(sigma)
    else:
        raise ValueError('Cost not implemented')
    
def const_cost(sigma_x, sigma_y,  cost='IGW', t=0.5):
    return polynomial_cost(sigma_x, t, cost=cost) + polynomial_cost(sigma_y, t, cost=cost)

def linearized_cost_matrix(sigma, cost='IGW', t=0.5):
    if cost == 'IGW':
        return sigma
    elif cost == 'DGW':
        l=len(sigma)
        return factorial(l-1)*matrix_cofactor_low_dim(sigma)
    elif cost == 'IDGW':
        l=len(sigma)
        return t*sigma+ (1-t)*factorial(l-1)*matrix_cofactor_low_dim(sigma)
    else:
        raise ValueError('Cost not implemented')
#########


def linearized_cost_function(space_x, space_y, M): 
    return (space_x @ M.T).dot(space_y.T)
   
def cross_covariance(space_x, space_y, pi):
    return np.einsum('kd,lD,kl->dD', space_x, space_y, pi)

def covariance(space_x, mu):
    return np.einsum('kd,kD,k->dD', space_x, space_x, mu)

def line_search_IGW(sigma_1, sigma_0):
    if np.linalg.norm(sigma_1)>np.linalg.norm(sigma_0):
            tau = 1
            T = np.linalg.norm(sigma_1)**2
    else :
            tau = 0
            T = np.linalg.norm(sigma_0)**2
    return T,tau

def line_search_IDGW_2d(sigma_1, sigma_0, t):
    d_1 = det_23d(sigma_1)
    d_0 = det_23d(sigma_0)
    tr = np.trace(sigma_1)*np.trace(sigma_0) - np.trace(sigma_1@sigma_0)
    n1 = np.sum(sigma_1**2)
    n0 = np.sum(sigma_0**2)
    alpha = d_1 + d_0 - tr
    beta = tr - 2*d_0
    gamma = d_0
    
    alpha = t*(n1+n0) + (1-t)*alpha
    beta = -2*t*n0 + (1-t)*beta 
    gamma = t*n0 + (1-t)*gamma
    
    def q(x):
        return alpha*x**2 + beta*x +gamma
    
    if alpha>0:
        if d_1>d_0:
                tau = 1
        else :
                tau = 0
    else:
        lbd = -beta/(2*alpha)
        tau = min(1,max(0,lbd))
    T = q(tau)
    return 2*T,tau

def line_search_DGW_2d(sigma_1, sigma_0):
    d_1 = det_23d(sigma_1)
    d_0 = det_23d(sigma_0)
    tr = np.trace(sigma_1)*np.trace(sigma_0) - np.trace(sigma_1@sigma_0)
    alpha = d_1 + d_0 - tr
    beta = tr - 2*d_0
    gamma = d_0
    
    def q(x):
        return alpha*x**2 + beta*x +gamma
    
    if alpha>0:
        if d_1>d_0:
                tau = 1
        else :
                tau = 0
    else:
        lbd = -beta/(2*alpha)
        tau = min(1,max(0,lbd))
    T = q(tau)
    return 2*T,tau

def line_search_DGW_3d(sigma_1, sigma_0):
    A= True
    
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
    
    alpha = d_1 - d_0 - TR_1 + TR_0
    beta = 3*d_0 + TR_1 - 2*TR_0
    gamma = -3*d_0 + TR_0
    delta = d_0
    
    def q(x):
        return alpha*x**3 + beta*x**2 + gamma*x + delta
    
    DELT = 4*beta**2 - 12*alpha*gamma
    if DELT>0:
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
    return 6*T,tau

def line_search_IDGW_3d(sigma_1, sigma_0, t):
    A= True
    
    d_1 = det_23d(sigma_1)
    d_0 = det_23d(sigma_0)
    n1 = np.sum(sigma_1**2)
    n0 = np.sum(sigma_0**2)
    
    tr_1 = np.trace(sigma_1)
    tr_0 = np.trace(sigma_0)
    tr_11 = np.trace(sigma_1@sigma_1)
    tr_00 = np.trace(sigma_0@sigma_0)
    tr_10 = np.trace(sigma_1@sigma_0)
    tr_110 = np.trace(sigma_1@sigma_1@sigma_0)
    tr_001 = np.trace(sigma_0@sigma_0@sigma_1)
    
    TR_0 = 1/2*((tr_1**2-tr_11)*tr_0) + tr_110 -tr_1*tr_10
    TR_1 = 1/2*((tr_0**2-tr_00)*tr_1) + tr_001 -tr_0*tr_10
    
    alpha = d_1 - d_0 - TR_1 + TR_0
    beta = 3*d_0 + TR_1 - 2*TR_0
    gamma = -3*d_0 + TR_0
    delta = d_0
    
    beta = t*(n1+n0) + (1-t)*beta
    gamma = -2*t*n0 + (1-t)*gamma
    delta = t*n0 + (1-t)*delta
    
    def q(x):
        return alpha*x**3 + beta*x**2 + gamma*x + delta
    
    DELT = 4*beta**2 - 12*alpha*gamma
    if DELT>0:
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
    return 6*T,tau



def line_search(sigma_1, sigma_0, cost='IGW', t=0.5):
    if cost == 'IGW':
        return line_search_IGW(sigma_1, sigma_0)
    elif  cost == 'DGW' and len(sigma_0)==2:
        return line_search_DGW_2d(sigma_1, sigma_0)
    elif  cost == 'DGW' and len(sigma_0)==3:
        return line_search_DGW_3d(sigma_1, sigma_0)
    elif cost == 'IDGW' and len(sigma_0)==2:
        return line_search_IDGW_2d(sigma_1, sigma_0, t)
    elif cost == 'IDGW' and len(sigma_0)==3:
        return line_search_IDGW_3d(sigma_1, sigma_0, t)
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
            
        M_pi_n = linearized_cost_matrix(sigma_pi_n, cost=cost, t=t)
        lin_cost = linearized_cost_function(space_x, space_y, M_pi_n)
        pi_n_1_hat = ot.emd(mu, nu, -lin_cost)
        pi_n_1_hat, log = ot.emd(mu, nu, M=-lin_cost, log=True)
        M_pi_val = -log['cost']
        sigma_pi_n_1_hat = cross_covariance(space_x, space_y, pi_n_1_hat)
        T, tau = line_search(sigma_pi_n_1_hat,sigma_pi_n,cost=cost, t=t) 
        print(f"FW it {it}: tau={tau}, {cost}^2 T cost={T}", end=' ')
        if tau == 0:
            break
        pi_n_1 = tau*pi_n_1_hat + (1-tau)*pi_n
        pi_n = pi_n_1
        yield pi_n_1_hat, M_pi_n, M_pi_val
    
    
def Frank_Wolfe(mu, space_x, nu, space_y, cost='IGW', pi_n=None, iter_max = 50, t=0.5):
    if pi_n is None:
        pi_n = np.outer(mu, nu)
    
    sigma_x = covariance(space_x, mu)
    sigma_y = covariance(space_y, nu)
    initial_cost = const_cost(sigma_x, sigma_y, cost=cost, t=t)
    print('initial_cost', initial_cost)
    for it in range(iter_max):
        sigma_pi_n = cross_covariance(space_x, space_y, pi_n)
        M_pi_n = linearized_cost_matrix(sigma_pi_n, cost=cost, t=t)
        lin_cost = linearized_cost_function(space_x, space_y, M_pi_n)
        pi_n_1_hat = ot.emd(mu, nu, -lin_cost)
        sigma_pi_n_1_hat = cross_covariance(space_x, space_y, pi_n_1_hat)
        T, tau = line_search(sigma_pi_n_1_hat, sigma_pi_n, cost=cost, t=t) 
        print(f"it {it}: tau={tau}, {cost}^2_cost={initial_cost-2*T}", end=' ')
        if tau == 0:
            break
        pi_n_1 = tau*pi_n_1_hat + (1-tau)*pi_n
        pi_n = pi_n_1
    return initial_cost-2*T, pi_n

def testing_2d():
    mu_a1, sigma_a = np.array([0.5, 0.5]), 0.3
    r_factor = 0.5
    mu_b, sigma_b = r_factor*mu_a1, 0.2
    n_grid_1d_x = 50
    n_grid_1d_y = 25
    def make_space_2d(n_grid):
        lin = np.linspace(-1, 1, n_grid)
        xx, yy = np.meshgrid(lin, lin)
        return np.vstack([xx.ravel(), yy.ravel()]).T
    space_x = make_space_2d(n_grid_1d_x)
    space_y = make_space_2d(n_grid_1d_y)

    factor = 0.5
    mu = factor*np.exp(-0.5 * (((space_x - mu_a1) / sigma_a) ** 2).sum(-1))
    mu += np.exp(-0.5 * (((space_x + factor*mu_a1) / sigma_a) ** 2).sum(-1))
    nu = factor*np.exp(-0.5 * (((space_y - mu_b) / sigma_b) ** 2).sum(-1))
    nu += np.exp(-0.5 * (((space_y + factor*mu_b) / sigma_b) ** 2).sum(-1))
    mu /= mu.sum()
    nu /= nu.sum()
    
    Frank_Wolfe(mu, space_x, nu, space_y, cost='IGW')
    Frank_Wolfe(mu, space_x, nu, space_y, cost='DGW')
    
def testing_3d():
    mu_a1, sigma_a = np.array([0.5, 0.5, 0.5]), 0.3
    r_factor = 0.5
    mu_b, sigma_b = r_factor*mu_a1, 0.2
    n_grid_1d_x = 15
    n_grid_1d_y = 12
    def make_space_3d(n_grid):
        lin = np.linspace(-1, 1, n_grid)
        xx, yy, zz = np.meshgrid(lin, lin, lin)
        return np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    space_x = make_space_3d(n_grid_1d_x)
    space_y = make_space_3d(n_grid_1d_y)

    factor = 0.5
    mu = factor*np.exp(-0.5 * (((space_x - mu_a1) / sigma_a) ** 2).sum(-1))
    mu += np.exp(-0.5 * (((space_x + factor*mu_a1) / sigma_a) ** 2).sum(-1))
    nu = factor*np.exp(-0.5 * (((space_y - mu_b) / sigma_b) ** 2).sum(-1))
    nu += np.exp(-0.5 * (((space_y + factor*mu_b) / sigma_b) ** 2).sum(-1))
    mu /= mu.sum()
    nu /= nu.sum()
    
    Frank_Wolfe(mu, space_x, nu, space_y, cost='IGW')
    Frank_Wolfe(mu, space_x, nu, space_y, cost='DGW')

if __name__ == "__main__":
    testing_2d()
    testing_3d()
 

   
