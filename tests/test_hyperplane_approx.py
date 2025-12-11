import numpy as np
import pytest
from xgw.Hyperplane_approx import *
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def marginals():
    mu_a1, sigma_a = np.array([0.5, 0.5]), 0.3
    r_factor = 0.5
    mu_b, sigma_b = r_factor*mu_a1, 0.2
    n_grid_1d_x = 10
    n_grid_1d_y = 10
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
    
    return mu, nu, space_x, space_y

def test_initial_box():
    mu, nu, space_x, space_y = marginals()
    e_base, R = construct_basis_eij(space_x, space_y)
    P_plus, P_minus = initial_box(e_base, mu, nu, emd_kwargs={})
    assert len(P_minus.V) == 8
    assert len(P_plus.V) == 16
    assert P_plus.H[0].shape == (8,4)
    assert P_minus.H[0].shape == (16,4)
    
    A,b = P_plus.H
    
    check = True
    for elem in P_minus.V:
        if not np.all(np.logical_or(A@elem<=b, np.isclose(A@elem,b))):
            logger.info(f'residual {A@elem-b}')
            check =  False
    assert check

    

@pytest.fixture
def niter():
    return 10


def test_overall(niter):
    mu, nu, space_x, space_y = marginals()
    x_1, x_2 = space_x.shape
    y_1, y_2 = space_y.shape
    logger.info(f'marginal points number : {x_1*x_2}, and {y_1*y_2}, dimension {2}')
    
    P_plus, P_minus, objective, previous_solutions_to_reuse = run_approx(mu, nu, space_x, space_y, emd_kwargs={'numItermax': 10**6}, niter = niter)
    _, _, new_obj, _ = Hausdorff( P_plus, P_minus, previous_solutions_to_reuse)
    
    logger.info(f'final  Hausdorf {new_obj - objective}')
    # computing Hausdorff distance
    # assert new_obj<=objective
    # checking P_minus is included in P_plus 
    # P_minus.V_to_H()
    # vertices_list = P_minus.V
    # A,b  = compute_polytope_halfspaces(vertices_list)
    A,b = P_plus.H
    check = True
    for i, elem in enumerate(P_minus.V):
        logger.info(f'Checking vertex {i}')
        if not np.all(np.logical_or(A@elem<=b, np.isclose(A@elem,b))):
            logger.info(f'residual {A@elem-b}')
            assert False
    
    return P_plus, P_minus, objective, previous_solutions_to_reuse

def projection_2d(niter):
    pass
    
    # P_plus, P_minus, objective, previous_solutions_to_reuse = test_overall()

# test_overall(10)
# # test_initial_box()
