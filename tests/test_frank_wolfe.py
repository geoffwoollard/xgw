import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from xgw.frank_wolfe import frank_wolfe_gw, center_marginal, optimize_deg_2_polynomial, optimize_deg_3_polynomial, dgw_2d_polynomial, det_23d, dgw_3d_polynomial, matrix_cofactor_low_dim
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
    
    alpha, beta, gamma = dgw_2d_polynomial(A, B)
    for p in np.linspace(0,1,10):
        assert np.allclose(2*det_23d(p*A + (1-p)*B), alpha*p**2 + beta*p + gamma)
    
    
    A = np.array([[1,0,0.0],[4,5,3], [2,6,1]])
    B = np.array([[5,2,4.0],[0,5,8], [0,0,1]])
    
    alpha, beta, gamma, delta = dgw_3d_polynomial(A, B)
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


def lie_group_action(space, M):
    return  space @ M


def random_rotation_matrix(d):
    random_angle = np.random.rand() * 2 * np.pi
    if d ==2:
        rotation = R.from_euler('z', random_angle).as_matrix()[:d,:d]
    elif d==3:
        random_axis = np.random.randn(3)
        random_axis /= np.linalg.norm(random_axis)
        rotation = R.from_rotvec(random_axis * random_angle).as_matrix()
    else:
        raise ValueError(f"dimension not implemented")
    return rotation


def random_invariance_matrix(cost, d):
    np.random.seed(2)
    if cost == 'IGW':
        rotation = random_rotation_matrix(d)
        s = np.random.randint(2, size = d)
        flips = (s == 1).astype(int) - (s == 0).astype(int)
        return np.diag(flips) @ rotation
    elif cost == 'CGW':
        rotation = random_rotation_matrix(d)
        return rotation
    elif cost == 'DGW':
        rotation = random_rotation_matrix(d)
        sheer_fact = 1+10*np.random.rand(d)
        sheer_fact /= np.power(np.prod(sheer_fact), 1/d)
        print (sheer_fact)
        return np.diag(sheer_fact) @ rotation
    else:
        raise ValueError(f"cost not implemented")


def compute_test(marg, cost, p):
    # marginals
    mu, nu, space_x, space_y = marg
    # Initial coupling
    pi_n = p*np.outer(mu, mu) + (1-p)*np.diag(mu) 
    # dimension
    d = space_x.shape[-1]
    # Non zero cost for two different marginals
    c, _, _ = frank_wolfe_gw(mu, space_x, nu, space_y, cost=cost)
    assert c>1e-3
    # Zero cost for the same marginals
    c, _, _ = frank_wolfe_gw(mu, space_x, mu, space_x, cost=cost, pi_n=pi_n)
    assert c<1e-15
    # invariance by Lie group action
    M = random_invariance_matrix(cost, d)
    c, _, _ = frank_wolfe_gw(mu, space_x, mu, lie_group_action(space_x, M), cost=cost, pi_n=pi_n)
    assert c<1e-15
    
    
def test_FW_diff_costs(marginals, marginals_3d, simple_marginals_2D):
    p = 0.9
    compute_test (marginals, 'IGW', p)
    compute_test (marginals_3d, 'IGW', p)
    compute_test (simple_marginals_2D, 'IGW', p)
    
    p = 0.9
    compute_test (marginals, 'DGW', p)
    compute_test (marginals_3d, 'DGW', p)
    compute_test (simple_marginals_2D, 'DGW', p)
    
    p = 0.9
    compute_test (marginals, 'CGW', p)
    compute_test (marginals_3d, 'CGW', p)
    compute_test (simple_marginals_2D, 'CGW', p)


def test_optimal_t_cste():
    # we have an upper bound for the highest eigenvalue of the determinant hessian (on the sphere):4. This is probabaly not tight, let's estimate the optimal upper bound
    def hess_mat(x):
        m = np.array([[0,0,0,0,x[8],-x[7],0,-x[5],x[4]],
                    [0,0,0,-x[8],0,x[6],x[5],0,-x[3]],
                    [0,0,0,x[7],-x[6],0,-x[4],x[3],0],
                    [0,0,0,0,0,0,0,x[2],-x[1]],
                    [0,0,0,0,0,0,-x[2],0,x[0]],
                    [0,0,0,0,0,0,x[1],-x[0],0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0]
                    ])
        return m+m.T
    
    def test_matrix_eig(niter):
        max_val = 0
        for iter in range(niter):
            x = np.random.rand(9)
            norm = np.linalg.norm(x)
            sign = np.random.randint(2, size = 9)
            sign = (sign == 1).astype(int) + (sign == -1).astype(int)
            eig = np.linalg.eigvalsh(hess_mat(x))
            e1, e2 = abs(eig[0]), abs(eig[-1])
            val = max(e1, e2)
            if max_val < val:
                max_val = val
        return max_val
    
    best_cst = test_matrix_eig(100000)
    assert 2.5<=best_cst<=3