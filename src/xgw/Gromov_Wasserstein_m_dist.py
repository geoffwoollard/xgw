import numpy as np
from numba import njit
import ot
from ncpol2sdpa import generate_variables, SdpRelaxation
from .Frank_Wolfe import _Frank_Wolfe_iter, covariance, const_cost, polynomial_cost, Frank_Wolfe_polynomial
from .Hyperplane_approx import run_approx, initial_box, projection, update_box, f_to_e, e_to_f, compute_hyperplane, function_to_cost
from .qp_incremental_projector import OptimalProjectedCoupling
import logging
logger = logging.getLogger(__name__)


    
def optimal_cost_cvx(P, cost, R, d, t):
    # optimize a convex polynomial cost over a polytope
    vect_list = np.array(P.V)
    return _optimal_cost_cvx(vect_list, cost, R, d, t)

def optimal_polynomial_cost(P, cost, relax_level, R, t):
    
    n_vars = len(R) # Number of variables
    # Requested level of relaxation relax_level
    x = generate_variables('x', n_vars) # polynomial variables
    obj = non_convex_polynomial_cost(cost, x, R, t) # polynomial cost to optimize
    (A,b) = P.H
    inequalities = list(np.ravel(-A@x+b)) # polytope inequalities
    # Relaxing and solving the problem
    sdp = SdpRelaxation(x)
    sdp.get_relaxation(relax_level, objective=obj, inequalities=inequalities)
    sdp.solve()
    
    x_op = np.array([sdp[x[i]] for i in range(n_vars)])
    c_op = sdp.primal
    logger.info(f'Polynomial optimization approximate solution : {sdp.status}, with optimal cost {c_op}')
    
    return c_op, x_op


def non_convex_polynomial_cost(cost, x, R, t):   
    l = len(x)
    x = x@R
    if  cost == 'DGW':
        if l==1:
            return x[0]
        elif l==4:
            return 2*(x[0]*x[3]-x[2]*x[1] )
        else:
            return 6*(x[0]*(x[4]*x[8]-x[7]*x[5]) - x[3]*(x[1]*x[8]-x[7]*x[2]) + x[6]*(x[1]*x[5]-x[4]*x[2]))

    elif cost == 'CGW':
        if l==1:
            return t*x[0]**2+(1-t)*x[0] 
        elif l==4:
            return t*(np.sum(np.dot(x, x))) + 2*(1-t)*(x[0]*x[3]-x[2]*x[1])
        else:
            return t*(np.sum(np.dot(x, x))) + 6*(1-t)*(x[0]*(x[4]*x[8]-x[7]*x[5]) - x[3]*(x[1]*x[8]-x[7]*x[2]) + x[6]*(x[1]*x[5]-x[4]*x[2]))
    else:
        raise ValueError('Cost not implemented')
    
@njit
def _optimal_cost_cvx(vect_list, cost, R, d, t):
    x_op = vect_list[0]
    c_op = vector_cost(x_op, cost, R, d, t)
    for vect in vect_list[1:]:
        c = vector_cost(vect, cost, R, d, t)
        if c > c_op:
            c_op = c
            x_op = vect
    return c_op, x_op

@njit
def vector_cost(vect, cost, R, d, t):
    # compute the cost of a vector after changing the base
    sigma = e_to_f(vect, R, d)
    return polynomial_cost(sigma, cost, t)



def vect_to_coupling(x_minus, mu, nu, e_base):
    l_mu = len(mu)
    l_nu = len(nu)
    dim = l_mu*l_nu
    proj_dim = len(x_minus) 
    A = -np.eye(dim)
    b = np.zeros(dim)
    # Need to check matrix C (marginal constraints), not exactly sure of the implementation
    C = np.zeros((l_mu, l_nu, l_mu + l_nu))
    for i in range(l_mu):
        C[i, :, i] = 1
    for j in range(l_nu):
        C[:, j, l_mu+j] = 1
    C = np.transpose(C.reshape((dim, l_mu + l_nu)))
    d = np.hstack(np.ravel(mu), np.ravel(nu))
    conv_solver = OptimalProjectedCoupling(dim, proj_dim, e_base, A=A, b=b, C=C, d=d)
    pi_opt, objective = conv_solver.solve(x_minus)
    # can do check on objective being close to 0 (to code later)
    pi_opt = pi_opt.reshape((l_mu,l_nu))
    return pi_opt
    
    

def GW_m_convex(mu, space_x, nu, space_y, emd_kwargs, cost='IGW', cost_tol=1e-5, iter_max=100, t=0.5):
    # This code is specifically designed for a convex cost, as IGW or CGW with a low enough t
    d = space_x.shape[-1]
    # Computing constant cost
    sigma_x = covariance(space_x, mu)
    sigma_y = covariance(space_y, nu)
    cst_cost = const_cost(sigma_x, sigma_y, cost, t)
    
    # Bounding box initialization
    e_base, R, P_plus, P_minus = initial_box(space_x, space_y, mu, nu, emd_kwargs)
    
    # Selection of the best direction (lagest score in the bounding box)
    c_plus, x_plus = optimal_cost_cvx(P_plus, cost, R, d, t)
    c_minus, x_minus = optimal_cost_cvx(P_minus, cost, R, d, t)
    
    centro = P_minus.get_centroid()
    g = x_plus - centro
    g /= np.linalg.norm(g)
    
    for iter in range(iter_max):
        g_hat, g_star = compute_hyperplane(mu, nu, g, e_base, emd_kwargs)
        P_plus, P_minus = update_box(P_plus, P_minus, [[g, g_hat]], [g_star])
        Tcost = vector_cost(g_star, cost, R, d, t)

        # The optimal values after each iteration is updated
        c_plus, x_plus = optimal_cost_cvx(P_plus, cost, R, d, t)
        
        if Tcost > c_minus:
            c_minus = Tcost
            x_minus = g_star
        if c_plus - c_minus < cost_tol:
            break
        
        # update direction
        centro = P_minus.get_centroid() # not needed after a couple of iterations
        g = x_plus - centro
        g /= np.linalg.norm(g)
    
    # Computing the optimal coupling:
    cost_matrix = function_to_cost(x_minus, e_base)
    pi= ot.emd(mu, nu, M=-cost_matrix, **emd_kwargs)
    
    # Local optimization to finish the optimization  (may not be needed)
    c_op, pi_opt = Frank_Wolfe_polynomial(mu, space_x, nu, space_y, pi, cost=cost, iter_max = 100, t=t)
    
    return cst_cost-2*c_op, pi_opt, c_plus - c_op

        

def GW_m_non_convex(mu, space_x, nu, space_y, emd_kwargs, relax_level=4, cost='IGW', cost_tol=1e-5, iter_max=100, FW_iter = 100, t=0.5):
    # Computing constant cost
    sigma_x = covariance(space_x, mu)
    sigma_y = covariance(space_y, nu)
    cst_cost = const_cost(sigma_x, sigma_y, cost, t)
    
    # Bounding box initialization
    e_base, R, P_plus, P_minus = initial_box(space_x, space_y, mu, nu, emd_kwargs)
    R_inv = np.linalg.inv(R)
    # Initialization of the cost bounds by optimizing non convex function over bounding boxes
    c_plus, x_plus = optimal_polynomial_cost(P_plus, cost, relax_level, R, t) #need implementation
    c_minus, x_minus = optimal_polynomial_cost(P_minus, cost, relax_level, R, t)
    centro = P_minus.get_centroid()
    init_direc = x_plus - centro
    init_direc/= np.linalg.norm(init_direc)
    for iter in range(iter_max):
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
        c_minus, x_minus = optimal_polynomial_cost(P_minus, cost, relax_level, R, t) 
        c_plus, x_plus = optimal_polynomial_cost(P_plus, cost, relax_level, R, t)
        if c_plus - c_minus < cost_tol:
            break
        # Choosing next direction 
        centro = P_minus.get_centroid()
        init_direc = x_plus - centro
        init_direc /= np.linalg.norm(init_direc)
    # Once the algorithm converges, the optimal point is x_minus, we find the appropriate transport plan
    pi = vect_to_coupling(x_minus, mu, nu, e_base) 
    # Local optimization 
    c_op, pi_opt = Frank_Wolfe_polynomial(mu, space_x, nu, space_y, pi, cost=cost, iter_max = FW_iter, t=t)
    
    return cst_cost-2*c_op, pi_opt, c_plus - c_minus


def GW_m_non_convex_Hausdorff(mu, space_x, nu, space_y, emd_kwargs, relax_level=4, cost='IGW', iter_max=100, FW_iter = 100, t=0.5):
    # Computing constant cost
    sigma_x = covariance(space_x, mu)
    sigma_y = covariance(space_y, nu)
    cst_cost = const_cost(sigma_x, sigma_y, cost, t)
    # Computing bounding box
    P_minus, Hausdorff_dist, R, e_base = run_approx(mu, nu, space_x, space_y, emd_kwargs, niter=iter_max, epsilon=1e-15)
    # Global optimization
    _, x_op = optimal_polynomial_cost(P_minus, cost, relax_level, R, t)
    # Computing coupling
    pi = vect_to_coupling(x_op, mu, nu, e_base)
    # Local optimization 
    c_op, pi_opt = Frank_Wolfe_polynomial(mu, space_x, nu, space_y, pi, cost=cost, iter_max = FW_iter, t=t)
    
    return cst_cost-2*c_op, pi_opt, Hausdorff_dist
    