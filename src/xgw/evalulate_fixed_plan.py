import numpy as np

from xgw.gromov_wasserstein_m_dist import gw_m_convex, vector_cost
from xgw.hyperplane_approx import construct_basis_eij, projection
from xgw.frank_wolfe import covariance, const_cost, center_marginal


def evaluate_cost_precompute(space_x, space_y, mu, nu, cost, t):
        # This code is specifically designed for a convex cost, as IGW or CGW with a high enough t
    d = space_x.shape[-1]
    # selecting the optimale t in the CGW case, if not pre-selected
    # t = optimal_t(max_diam, d, cost, t, convex_tol)
    
    space_x, space_y = center_marginal(mu, space_x, nu, space_y)
    # Computing constant cost
    sigma_x = covariance(space_x, mu)
    sigma_y = covariance(space_y, nu)
    cst_cost = const_cost(sigma_x, sigma_y, cost, t)
    e_base, R = construct_basis_eij(space_x, space_y)
    return cst_cost, e_base, R, d, t

def evaluate_cost_postcompute(e_base, R, d, t, plan, cost, cst_cost):
    g_star = projection(plan, e_base)
    c_opt = vector_cost(g_star, cost, R, d, t)
    total_cost = cst_cost-2*c_opt
    return total_cost

def evaluate_cost(space_x, space_y, plan, cost, t):
    mu = plan.sum(axis=1)
    nu = plan.sum(axis=0)
    cst_cost, e_base, R, d, t = evaluate_cost_precompute(space_x, space_y, mu, nu, cost, t)
    total_cost = evaluate_cost_postcompute(e_base, R, d, t, plan, cost, cst_cost)
    return total_cost

# def evaluate_cost(space_x, space_y, plan, cost, t):
#         # This code is specifically designed for a convex cost, as IGW or CGW with a high enough t
#     d = space_x.shape[-1]
#     mu = plan.sum(axis=1)
#     nu = plan.sum(axis=0)
#     # selecting the optimale t in the CGW case, if not pre-selected
#     # t = optimal_t(max_diam, d, cost, t, convex_tol)
    
#     space_x, space_y = center_marginal(mu, space_x, nu, space_y)
#     # Computing constant cost
#     sigma_x = covariance(space_x, mu)
#     sigma_y = covariance(space_y, nu)
#     cst_cost = const_cost(sigma_x, sigma_y, cost, t)

#     e_base, R = construct_basis_eij(space_x, space_y)
#     g_star = projection(plan, e_base)
#     c_opt = vector_cost(g_star, cost, R, d, t)
#     total_cost = cst_cost-2*c_opt
#     return total_cost

def make_simple_marginals(seed, d, min_points=30, max_points=30):
    np.random.seed(seed)
    n_points = np.random.randint(min_points, max_points+1)
    print(f'Number of points in simple marginals test: {n_points}')
    mu = nu = np.ones(n_points) / n_points

    space_x = np.random.randn(n_points,d)
    space_y = np.random.randn(n_points,d)
    radius_x = np.linalg.norm(space_x, axis=1).max()
    radius_y = np.linalg.norm(space_y, axis=1).max()
    space_x /= radius_x
    space_y /= radius_y

    return mu, nu, space_x, space_y

def make_marginals(seed, min_points=5, max_points=15):
    np.random.seed(seed)
    n_points_xy = np.random.randint(min_points, max_points+1)
    mu_a1, sigma_a = np.array([0.5, 0.5]), 0.3
    r_factor = 0.5
    mu_b, sigma_b = r_factor*mu_a1, 0.2
    n_grid_1d_x = n_grid_1d_y = n_points_xy
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

    # scale space to have zero center of mass
    space_x = space_x.astype(mu.dtype)
    space_y = space_y.astype(nu.dtype)
    space_x -= (space_x * mu[:, None]).sum(axis=0)
    space_y -= (space_y * nu[:, None]).sum(axis=0)
    
    return mu, nu, space_x, space_y

