import numpy as np
import pytest
from xgw.Frank_Wolfe import Frank_Wolfe_GW, center_marginal, optimize_deg_2_polynomial, optimize_deg_3_polynomial, DGW_2d_polynomial, det_23d, DGW_3d_polynomial, matrix_cofactor_low_dim
from test_hyperplane_approx import marginals,  simple_marginals_2D


def test_polynomial():
    T, tau = optimize_deg_2_polynomial(-1, 1, 0) #optimizing x(1-x)
    assert np.allclose(1/2, tau) and np.allclose(1/4, T)
    T, tau = optimize_deg_2_polynomial(-1, 2, 0) #optimizing x(2-x)
    assert np.allclose(1, tau) and np.allclose(1, T)
    T, tau = optimize_deg_2_polynomial(1, -2, 4) #optimizing (x-1)^2 +3
    assert np.allclose(0, tau) and np.allclose(4, T)
    
    T, tau = optimize_deg_3_polynomial(1/3, -9/4, 2, 4) 
    assert np.allclose(1/2, tau) 
    T, tau = optimize_deg_3_polynomial(1/3, 0, 0, 0) 
    assert np.allclose(1, tau) 
    T, tau = optimize_deg_3_polynomial(1, -0.2, 0, 0) 
    assert np.allclose(1, tau) 
    T, tau = optimize_deg_3_polynomial(1, -10, 0, 0) 
    assert np.allclose(0, tau) 



def test_determinant_interpolation():
    A = np.array([[1,0.0],[4,5]])
    B = np.array([[5,2.0],[0,5]])
    
    alpha, beta, gamma = DGW_2d_polynomial(A, B)
    for p in np.linspace(0,1,10):
        assert np.allclose(2*det_23d(p*A + (1-p)*B), alpha*p**2 + beta*p + gamma)
    
    
    A = np.array([[1,0,0.0],[4,5,3], [2,6,1]])
    B = np.array([[5,2,4.0],[0,5,8], [0,0,1]])
    
    alpha, beta, gamma, delta = DGW_3d_polynomial(A, B)
    for p in np.linspace(0,1,10):
        assert np.allclose(det_23d(p*A + (1-p)*B), np.linalg.det(p*A + (1-p)*B))
        assert np.allclose(6*det_23d(p*A + (1-p)*B), alpha*p**3 + beta*p**2 + gamma*p + delta)
        
def test_comatrix():
    A = np.array([[1,0,0],[4,5,3], [2,6,1]])
    invA = 1/det_23d(A)*np.transpose(matrix_cofactor_low_dim(A))
    real_inv = np.linalg.inv(A)
    print(invA, real_inv)
    assert np.allclose(invA, real_inv)
    A = np.array([[5,2,4],[0,5,8], [0,0,1]])
    invA = 1/det_23d(A)*np.transpose(matrix_cofactor_low_dim(A))
    real_inv = np.linalg.inv(A)
    print(invA, real_inv)
    assert np.allclose(invA, real_inv)
    
    
    
@pytest.fixture
def marginals_3d():
    mu_a1, sigma_a = np.array([0.5, 0.5, 0.5]), 0.3
    r_factor = 0.5
    mu_b, sigma_b = r_factor*mu_a1, 0.2
    n_grid_1d_x = 10
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
    return mu, nu, space_x, space_y

def test_centered_marginals(simple_marginals_2D):
    mu, nu, space_x, space_y = simple_marginals_2D
    c_mu = (space_x * mu[:, None]).sum(axis=0)
    c_nu = (space_y * nu[:, None]).sum(axis=0)
    
    assert not np.allclose(c_mu,0) and not np.allclose(c_nu,0)
    space_x, space_y = center_marginal(mu, space_x, nu, space_y)
    
    c_mu = (space_x * mu[:, None]).sum(axis=0)
    c_nu = (space_y * nu[:, None]).sum(axis=0)
    assert np.allclose(c_mu,0) and np.allclose(c_nu,0)

    
    
def test_IGW(marginals, marginals_3d, simple_marginals_2D):
    # 2D marginals
    mu, nu, space_x, space_y = marginals
    
    c, _, _ = Frank_Wolfe_GW(mu, space_x, nu, space_y, cost='IGW')
    assert c>1e-3

    p = 0.99999
    pi_n = p*np.outer(mu, mu) + (1-p)*np.diag(mu) 
    c, _, _ = Frank_Wolfe_GW(mu, space_x, mu, space_x, cost='IGW', pi_n=pi_n)
    assert c<1e-15
    
    
    # 3D marginals
    mu, nu, space_x, space_y = marginals_3d
    
    c, _, _ = Frank_Wolfe_GW(mu, space_x, nu, space_y, cost='IGW')
    assert c>1e-3

    p = 0.99999
    pi_n = p*np.outer(mu, mu) + (1-p)*np.diag(mu) 
    c, _, _ = Frank_Wolfe_GW(mu, space_x, mu, space_x, cost='IGW', pi_n=pi_n)
    assert c<1e-15
    
    
    # simple 2D marginals
    mu, nu, space_x, space_y = simple_marginals_2D
    
    c, _, _ = Frank_Wolfe_GW(mu, space_x, nu, space_y, cost='IGW')
    assert c>1e-3

    p = 0.99999
    pi_n = p*np.outer(mu, mu) + (1-p)*np.diag(mu) 
    c, _, _ = Frank_Wolfe_GW(mu, space_x, mu, space_x, cost='IGW', pi_n=pi_n)
    assert c<1e-15
    
    
#     #Add more tests here
    
    


def test_DGW(marginals, marginals_3d, simple_marginals_2D):
    # 2D marginals
    mu, nu, space_x, space_y = marginals
    
    c, _, _ = Frank_Wolfe_GW(mu, space_x, nu, space_y, cost='DGW')
    assert c>1e-3

    for p in np.linspace(0,1,20):
        # it looks like there is a problem here, sometimes the algorithm stops at first step
        print(f'p = {p}')
        pi_n = p*np.outer(mu, mu) + (1-p)*np.diag(mu) 
        c, _ , initial_coupling_cost = Frank_Wolfe_GW(mu, space_x, mu, space_x, cost='DGW', pi_n=pi_n)
        assert initial_coupling_cost - c > 0 or c < 1e-15

    p = 0.99
    pi_n = p*np.outer(mu, mu) + (1-p)*np.diag(mu) 
    c, _, _ = Frank_Wolfe_GW(mu, space_x, mu, space_x, cost='DGW', pi_n=pi_n)
    assert c<1e-15
    
#     # 3D marginals to fix
    mu, nu, space_x, space_y = marginals_3d
    
    c, _, _ = Frank_Wolfe_GW(mu, space_x, nu, space_y, cost='DGW')
    assert c>1e-3

    for p in np.linspace(0,1,20):
        # it looks like there is a problem here, sometimes the algorithm stops at first step
        print(f'p = {p}')
        pi_n = p*np.outer(mu, mu) + (1-p)*np.diag(mu) 
        c, _ , initial_coupling_cost = Frank_Wolfe_GW(mu, space_x, mu, space_x, cost='DGW', pi_n=pi_n)
        assert initial_coupling_cost - c > 0 or c < 1e-15
    
    p = 0.99
    pi_n = p*np.outer(mu, mu) + (1-p)*np.diag(mu) 
    c, _, _ = Frank_Wolfe_GW(mu, space_x, mu, space_x, cost='DGW', pi_n=pi_n)
    assert c<1e-15
    
    # simple 2D marginals
    mu, nu, space_x, space_y = simple_marginals_2D
    
    c, _, _ = Frank_Wolfe_GW(mu, space_x, nu, space_y, cost='DGW')
    assert c>1e-3

    p = 0.99
    pi_n = p*np.outer(mu, mu) + (1-p)*np.diag(mu) 
    c, _, _ = Frank_Wolfe_GW(mu, space_x, mu, space_x, cost='DGW', pi_n=pi_n)
    assert c<1e-15
    
    
    #Add more tests here


