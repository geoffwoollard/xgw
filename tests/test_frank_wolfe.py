import numpy as np
import logging
from xgw.Frank_Wolfe import Frank_Wolfe_GW





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
    
    c, _ = Frank_Wolfe_GW(mu, space_x, nu, space_y, cost='IGW')
    assert c>1e-3
    c, _ = Frank_Wolfe_GW(mu, space_x, nu, space_y, cost='DGW')
    assert c>1e-3
    c, _ = Frank_Wolfe_GW(mu, space_x, nu, space_y, cost='CGW', t= 0.5)
    assert c>1e-3
    c, _ = Frank_Wolfe_GW(mu, space_x, mu, space_x, cost='IGW')
    assert c<1e-15
    c, _ = Frank_Wolfe_GW(mu, space_x, mu, space_x, cost='DGW')
    assert c<1e-15
    c, _ = Frank_Wolfe_GW(mu, space_x, mu, space_x, cost='CGW', t= 0.5)
    assert c<1e-15
    
def testing_3d():
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
    
    c, _ = Frank_Wolfe_GW(mu, space_x, nu, space_y, cost='IGW')
    assert c>1e-2
    c, _ = Frank_Wolfe_GW(mu, space_x, nu, space_y, cost='DGW')
    assert c>1e-2
    c, _ = Frank_Wolfe_GW(mu, space_x, nu, space_y, cost='CGW', t= 0.5)
    assert c>1e-2
    
    pi_n = 0.1*np.outer(mu, mu) + 0.9*np.diag(mu)
    
    c, _ = Frank_Wolfe_GW(mu, space_x, mu, space_x, cost='IGW', pi_n=pi_n)
    assert c<1e-10
    c, _ = Frank_Wolfe_GW(mu, space_x, mu, space_x, cost='DGW', pi_n=pi_n)
    assert c<1e-10
    c, _ = Frank_Wolfe_GW(mu, space_x, mu, space_x, cost='CGW', t= 0.5, pi_n=pi_n)
    assert c<1e-10
    