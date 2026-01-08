import numpy as np
from numba import njit
import ot
from .Frank_Wolfe import _Frank_Wolfe_iter, covariance, const_cost, polynomial_cost
from .Hyperplane_approx import run_approx, initial_box, projection, update_box, f_to_e, e_to_f, compute_hyperplane, function_to_cost


def optimal_cost_cvx(P, cost, R_inv, t=0.5):
    # optimize a convex polynomial cost over a polytope
    vect_list = np.array(P.V)
    return _optimal_cost_cvx(vect_list, cost, R_inv, t=t)

@njit
def _optimal_cost_cvx(vect_list, cost, R_inv, t=0.5):
    x_op = vect_list[0]
    c_op = vector_cost(x_op, cost, R_inv, t=t)
    for vect in vect_list[1:]:
        c = vector_cost(vect, cost, R_inv, t=t)
        if c > c_op:
            c_op = c
            x_op = vect
    return c_op, x_op

@njit
def vector_cost(vect, cost, R_inv, t=0.5):
    # compute the cost of a vector after changing the base
    sigma = e_to_f(vect, R_inv)
    return polynomial_cost(sigma, cost, t)

def GW_m_convex(mu, space_x, nu, space_y, emd_kwargs, cost='IGW', cost_tol=1e-5, iter_max=10000, t=0.5):
    # This code is specifically designed for a convex cost
    
    # Computing constant cost
    sigma_x = covariance(space_x, mu)
    sigma_y = covariance(space_y, nu)
    cst_cost = const_cost(sigma_x, sigma_y, cost, t)
    
    # Bounding box initialization
    e_base, R, P_plus, P_minus = initial_box(space_x, space_y, mu, nu, emd_kwargs)
    R_inv = np.linalg.inv(R)
    
    # Selection of the best direction (lagest score in the bounding box)
    c_plus, x_plus = optimal_cost_cvx(P_plus, cost, R_inv, t=t)
    c_minus, x_minus = optimal_cost_cvx(P_minus, cost, R_inv, t=t)
    
    centro = P_minus.get_centroid()
    g = x_plus - centro
    g /= np.linalg.norm(g)
    
    for iter in range(iter_max):
        g_hat, g_star = compute_hyperplane(mu, nu, g, e_base, emd_kwargs)
        P_plus, P_minus = update_box(P_plus, P_minus, [[g, g_hat]], [g_star])
        Tcost = vector_cost(g_star, cost, R_inv, t=t)

        # The optimal values after each iteration is updated
        c_plus, x_plus = optimal_cost_cvx(P_plus, cost, R, t=t)
        if Tcost > c_minus:
            c_minus = Tcost
            x_minus = g_star
        if c_plus - c_minus < cost_tol:
            break
    
    # Computing the optimal coupling:
    cost_matrix = function_to_cost(x_minus, e_base)
    pi_opt = ot.emd(mu, nu, M=-cost_matrix, **emd_kwargs)
    
    return cst_cost-2*c_plus, pi_opt
        

def GW_m_non_convex(mu, space_x, nu, space_y, emd_kwargs, cost='IGW', cost_tol=1e-5, FW_iter_max=100, t=0.5):
    # Computing constant cost
    sigma_x = covariance(space_x, mu)
    sigma_y = covariance(space_y, nu)
    cst_cost = const_cost(sigma_x, sigma_y, cost, t)
    
    # Bounding box initialization
    e_base, R, P_plus, P_minus = initial_box(space_x, space_y, mu, nu, emd_kwargs)
    R_inv = np.linalg.inv(R)
    # Initialization of the cost bounds by optimizing non convex function over bounding boxes
    c_plus, x_plus = optimal_polynomial_cost(P_plus, cost, R_inv, t) #need implementation
    c_minus, x_minus = optimal_polynomial_cost(P_minus, cost, R_inv, t)
    centro = P_minus.get_centroid()
    init_direc = x_plus - centro
    init_direc/= np.linalg.norm(init_direc)
    for iter in range(FW_iter_max):
        # constraints that will be added to the bounding bow
        vertex_list = []
        half_plans_list = []
        for coup_star, g, g_hat in _Frank_Wolfe_iter(mu, space_x, nu, space_y, init_direc, cost=cost, t=t):
            # during the iteration, the OT problem (10) is solved with direction g and cost g_hat
            g_star = projection(coup_star, e_base)
            # g is in the f basis not in the e basis, and with a wrong format
            g = f_to_e(g, R_inv)
            vertex_list.append(g_star)
            half_plans_list.append([g, g_hat])
        # The bounding boxes are updated 
        P_plus, P_minus = update_box(P_plus, P_minus, half_plans_list, vertex_list)
        # The optimal values after each FW loops are updated
        c_minus, x_minus = optimal_polynomial_cost(P_minus, cost, R_inv, t) 
        c_plus, x_plus = optimal_polynomial_cost(P_plus, cost, R_inv, t)
        if c_plus - c_minus < cost_tol:
            break
        # Choosing next direction 
        centro = P_minus.get_centroid()
        init_direc = x_plus - centro
        init_direc /= np.linalg.norm(init_direc)
    # Once the algorithm converges, the optimal point is x_minus, we find the appropriate transport plan
    
    pi_opt = vect_to_coupling(x_minus, mu, nu) #need implementation
    
    return cst_cost-2*c_minus, pi_opt