import numpy as np
import pytest
import matplotlib.pyplot as plt
from xgw.Hyperplane_approx import *
import logging
from itertools import permutations
from pypoman.polygon import compute_polygon_hull

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@pytest.fixture
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


def test_initial_box(marginals):
    mu, nu, space_x, space_y = marginals
    e_base, R = construct_basis_eij(space_x, space_y)
    P_plus, P_minus = initial_box(e_base, mu, nu, emd_kwargs={})
    assert len(P_minus.V) == 8
    assert len(P_plus.V) == 16
    assert P_plus.H[0].shape == (8,4)
    assert P_minus.H[0].shape == (16,4)
    
    residuals = P_plus_outside_P_minus(P_plus, P_minus)
    atol = 1e-15
    assert np.all(residuals <= atol), f'max residual for inclusion of P_minus in P_plus : {residuals.max()}'


def simple_marginals_2D():
    mu = np.array([1/3,1/3,1/3])
    nu = mu = np.array([1/3,1/3,1/3])
    space_x = np.array([[0,1.53],[4.87,1],[5,1/2]])
    space_y = np.array([[3,0],[4,2],[1,2]])
    return mu, nu, space_x, space_y

@pytest.fixture
def niter():
    return 10

def permut_matrix(permut, dim):
    sol = np.zeros((dim,dim))
    for i,j in enumerate(permut):
        sol[i,j] = 1
    return sol
        

def test_simple_marginals(niter):
    mu, nu, space_x, space_y = simple_marginals_2D()
    x_1, x_2 = space_x.shape
    y_1, y_2 = space_y.shape
    print(f'marginal points number : {x_1*x_2}, and {y_1*y_2}', f'dimension {2}')
    
    P_plus, P_minus, objective, previous_solutions_to_reuse, objective_list = run_approx(mu, nu, space_x, space_y, emd_kwargs={}, niter = niter)
    logger.info(f'Objective list over iterations: {objective_list}')
    diffs = np.diff(objective_list)
    tol = 1e-16
    assert np.all(diffs < tol), f'Objective not non-increasing, diffs: {diffs}'
    _, _, new_obj, _ = Hausdorff( P_plus, P_minus, {})
    
    print(f'final  Hausdorf {new_obj}')
    assert new_obj<=objective
    residuals = P_plus_outside_P_minus(P_plus, P_minus)
    atol = 1e-20
    assert np.all(residuals <= atol)
    logger.info(f'max residual for inclusion of P_minus in P_plus : {residuals.max()}')

    
    e_base, _ = construct_basis_eij(space_x, space_y)
    coupling_vertices = [permut_matrix(elem, 3)/3 for elem in permutations([0,1,2])] # the true vertices of the coupling polytopes are the permutation matrices

    projected_couplings = [projection(vertex, e_base) for vertex in coupling_vertices]
    
    # ploting the projection in 2d of P_minus, projected_vertices and  P_plus
    P_minus_proj = np.array(P_minus.V)
    P_plus_proj = np.array(P_plus.V)
    
    P_true_proj_2d = DoubleRepresentation()
    P_true_proj_2d.add_V (np.array(projected_couplings)[:,:2])
    A, b = P_true_proj_2d.H
    true_vertices = np.array(compute_polygon_hull(A, b))

    plt.figure()
    plt.title('2D projection of the 4D spaces')
    plt.fill(true_vertices[:,0],true_vertices[:,1],  label=r'$P_{\Pi}$', alpha=0.3 )
    plt.scatter(P_plus_proj[:,0], P_plus_proj[:,1],c='k', label=r'$P_{\Pi}^+$')
    plt.scatter(P_minus_proj[:,0], P_minus_proj[:,1],c='r', label=r'$P_{\Pi}^-$')
    plt.legend()
    plt.savefig('img/test_projection_coupling_P_plus_minus')
    plt.close()
    
    
    P_true = DoubleRepresentation()
    P_true.add_V (projected_couplings)
    P_true.H_to_V()
    
    # checking P_minus is included in P_plus 
    A,b = P_plus.H
    check = True
    for i, elem in enumerate(P_minus.V):
        logger.info(f'Checking vertex {i}')
        if not np.all(A@elem<=b+tol):
            logger.info(f'residual {A@elem-b}')
            check =  False
    assert check, f'P_minus no included in P_plus'
    
    # checking P_minus is included in P_true 
    A,b = P_true.H
    check = True
    for i, elem in enumerate(P_minus.V):
        logger.info(f'Checking vertex {i}')
        if not np.all(A@elem<=b+tol):
            logger.info(f'residual {A@elem-b}')
            print (A@elem-b)
            check =  False
    assert check, f'P_minus no included in P_true'
    
    # checking P_true is included in P_plus 
    A,b = P_plus.H
    check = True
    for i, elem in enumerate(P_true.V):
        logger.info(f'Checking vertex {i}')
        if not np.all(A@elem<=b+tol):
            logger.info(f'residual {A@elem-b}')
            check =  False
    assert check, f'P_true no included in P_plus'
    
@pytest.fixture
def niter():
    return 5


def test_overall(niter, marginals):
    mu, nu, space_x, space_y = marginals
    x_1, x_2 = space_x.shape
    y_1, y_2 = space_y.shape
    logger.info(f'marginal points number : {x_1*x_2}, and {y_1*y_2}, dimension {2}')
    
    P_plus, P_minus, objective, previous_solutions_to_reuse, objective_list = run_approx(mu, nu, space_x, space_y, emd_kwargs={'numItermax': 10**6}, niter = niter)
    logger.info(f'Objective list over iterations: {objective_list}')
    diffs = np.diff(objective_list)
    tol = 1e-16

    assert np.all(diffs < tol), f'Objective not non-increasing, diffs: {diffs}'
    _, _, new_obj, _ = Hausdorff( P_plus, P_minus, previous_solutions_to_reuse)
    
    logger.info(f'final  Hausdorf {new_obj - objective}')
    assert new_obj<=objective

    residuals = P_plus_outside_P_minus(P_plus, P_minus)
    atol = 1e-17
    assert np.all(residuals <= atol), f'max residual for inclusion of P_minus in P_plus : {residuals.max()}'

