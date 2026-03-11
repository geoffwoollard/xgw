import numpy as np
from numba import njit
import ot
from ncpol2sdpa import generate_variables, SdpRelaxation
from .frank_wolfe import _frank_wolfe_iter, covariance, const_cost, polynomial_cost, frank_wolfe_polynomial, center_marginal, _classical_gw_frank_wolfe, classical_gw_const_cost
from .hyperplane_approx import run_approx, initial_box, projection, update_box, f_to_e, e_to_f, compute_hyperplane, classical_gw_initial_box
from .qp_incremental_projector import OptimalProjectedCoupling
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=False,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
    
def optimal_cost_cvx(P, cost, R, d, t):
    # optimize a convex polynomial cost over a polytope
    vect_list = np.array(P.V)
    return _optimal_cost_cvx(vect_list, cost, R, d, t)

@njit
def classical_gw_optimal_cost(vect_list, R):
    x_op = vect_list[0]
    vect_r = x_op@R
    c_op = 2*np.sum(vect_r[:-1]**2)+vect_r[-1]
    for i in range(len(vect_list)):
        vect = vect_list[i]
        vect_r = vect@R
        c = 2*np.sum(vect_r[:-1]**2)+vect_r[-1]
        if c > c_op:
            c_op = c
            x_op = vect
    return c_op, x_op

def optimal_polynomial_cost(P, cost, relax_level, R, t):
    
    n_vars = len(R) # Number of variables
    # Requested level of relaxation relax_level
    x = generate_variables('x', n_vars) # polynomial variables
    obj = non_convex_polynomial_cost(cost, x, R, t) # polynomial cost to optimize
    (A,b) = P.H
    inequalities = list(np.ravel(-A@x+b)) # polytope inequalities
    # Relaxing and solving the problem
    logger.info('Starting polynomial optimization over polytope with %d inequalities and %d variables at relaxation level %d', len(inequalities), n_vars, relax_level)
    logger.info(f'Polytope has  : {len(P.V)}, vertices')
    sdp = SdpRelaxation(x)
    logger.info('Generating relaxation...')
    sdp.get_relaxation(relax_level, objective=obj, inequalities=inequalities)
    logger.info('Solving relaxation...')
    sdp.solve()
    logger.info('Relaxation solved.')
    
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
    for i in range(len(vect_list)):
        vect = vect_list[i]
        c = vector_cost(vect, cost, R, d, t)
        if c > c_op:
            c_op = c
            x_op = vect
    return c_op, x_op

@njit
def vector_cost(vect, cost, R, d, t):
    '''Compute the cost of a vector after changing the base.'''
    sigma = e_to_f(vect, R, d)
    return polynomial_cost(sigma, cost, t)


def vect_to_coupling(x_minus, mu, nu, e_base):
    proj_dim = len(x_minus) 
    conv_solver = OptimalProjectedCoupling(proj_dim, e_base, mu=mu, nu=nu)
    pi_opt, objective = conv_solver.solve(x_minus)
    # can do check on objective being close to 0 (to code later)
    pi_opt = pi_opt.reshape((len(mu),len(nu)))
    return pi_opt
    
    
def optimal_t(max_diam, d, cost, t, convex_tol, exp_upper_bound=True):
    if cost == 'CGW' and t is None:
        if d <= 2 :
            t =  1/2 + convex_tol
        elif d == 3:
            assert max_diam is not None, f'Need to define a max_diameter or a t in 3D to check CGW convexity'
            if exp_upper_bound:
                val = (8*max_diam**2)/(2+8*max_diam**2)
            else:
                val = (24*max_diam**2)/(2+24*max_diam**2)
            if 1-val> 2*convex_tol:
                t = val + convex_tol
            else:
                t = (1+val)/2
    if t is None:
        return 0
    return t 


def new_direction_convex_slow(P_minus, x_plus, tol=1e-10):
    # Normalized normal vectors of P_minus
    A,b =  P_minus.H
    # Finding best direction 
    testing_dir = A @ x_plus - b
    index = np.argmax(testing_dir)
    g = A[index]
    # checking that the point is outside P_minus
    assert g @ x_plus - b[index]>= -tol 
    return g / np.linalg.norm(g)

def gw_m_convex(mu, space_x, nu, space_y, emd_kwargs, cost='IGW', gap_tol=1e-5, iter_max=100, FW_iter=100, t=None, max_diam=None, convex_tol = 1e-3, p_plus_implementation='cdd'):

    # This code is specifically designed for a convex cost, as IGW or CGW with a high enough t
    d = space_x.shape[-1]
    # selecting the optimale t in the CGW case, if not pre-selected
    t = optimal_t(max_diam, d, cost, t, convex_tol)
    space_x, space_y = center_marginal(mu, space_x, nu, space_y)
    # Computing constant cost
    sigma_x = covariance(space_x, mu)
    sigma_y = covariance(space_y, nu)
    cst_cost = const_cost(sigma_x, sigma_y, cost, t)
    
    # Bounding box initialization
    e_base, R, P_plus, P_minus = initial_box(space_x, space_y, mu, nu, emd_kwargs, p_plus_implementation=p_plus_implementation)
    # print('Initial box with ', len(P_minus.V), ' vertices and ', P_minus.H[0].shape[0], ' half-planes.')
    # print('Vertices: ', P_minus.V)
    
    # Selection of the best direction (lagest score in the bounding box)
    c_plus, x_plus = optimal_cost_cvx(P_plus, cost, R, d, t)
    c_minus, x_minus = optimal_cost_cvx(P_minus, cost, R, d, t)
    
    
    for iter in range(iter_max):
        # choose direction
        logger.info('Chosing best direction')
        g = new_direction_convex_slow(P_minus, x_plus)
        logger.info(f'Computing hyperhplane for direction g: {g}')
        g_hat, g_star = compute_hyperplane(mu, nu, g, e_base, emd_kwargs)
        logger.info(f'new half plane: {g}, {g_hat}')
        logger.info(f'new vertex: {g_star}')
        logger.info('Updating bounding boxes')
        P_plus, P_minus = update_box(P_plus, P_minus, [[g, g_hat]], [g_star])
        logger.info(f'Updated box with , P_plus {len(P_plus.V)} P_minus {len(P_minus.V)} vertices.')
        logger.info('Updating c_minus (candidate optimal value)')
        Tcost = vector_cost(g_star, cost, R, d, t)

        # The optimal values after each iteration is updated
        logger.info('Updating c_plus and c_minus')
        c_plus, x_plus = optimal_cost_cvx(P_plus, cost, R, d, t)
        logger.info(f'Iteration {iter}: c_minus={c_minus:1.20f}, c_plus={c_plus:1.20f}, c_plus - c_minus = {(c_plus - c_minus):1.20f}, vertices in P_minus {len(P_minus.V)} and P_plus {len(P_plus.V)}, half-spaces in P_minus {P_minus.H[0].shape[0]} and P_plus {P_plus.H[0].shape[0]}')
        
        if Tcost > c_minus:
            c_minus = Tcost
            x_minus = g_star
            
        if c_plus - c_minus < gap_tol:
            x_minus = g_star
            break

    # Computing the optimal coupling:
    pi_opt = vect_to_coupling(x_minus, mu, nu, e_base) 
    
    # Local optimization to finish the optimization  (may not be needed)
    c_op, pi_opt = frank_wolfe_polynomial(mu, space_x, nu, space_y, pi_opt, cost=cost, iter_max=FW_iter, t=t)

    total_loss = cst_cost-2*c_op
    gap = c_plus - c_op
    lower_bound_on_total_loss = cst_cost-2*c_plus
    logger.info(f'Final results: total loss {total_loss}, gap {gap}, bound on loss {lower_bound_on_total_loss}')

    
    return total_loss, pi_opt, gap, lower_bound_on_total_loss, cst_cost


def classical_gw(mu, space_x, nu, space_y, emd_kwargs, cost_tol=1e-5, iter_max=100, FW_iter=100, p_plus_implementation='cdd'):

    # This code is specifically designed for a convex cost, as IGW or CGW with a high enough t
    d = space_x.shape[-1]
    space_x, space_y = center_marginal(mu, space_x, nu, space_y)
    # Computing constant cost
    g_func, cst_cost = classical_gw_const_cost(mu, space_x, nu, space_y)
    # Bounding box initialization
    e_base, R, P_plus, P_minus = classical_gw_initial_box(space_x, space_y, g_func, mu, nu, emd_kwargs, p_plus_implementation=p_plus_implementation)

    # Selection of the best direction (lagest score in the bounding box)
    c_plus, x_plus = classical_gw_optimal_cost(np.array(P_plus.V), R)
    c_minus, x_minus = classical_gw_optimal_cost(np.array(P_minus.V), R)
    
    
    for iter in range(iter_max):
        
        # choose direction
        logger.info('Chosing best direction')
        g = new_direction_convex_slow(P_minus, x_plus)
        logger.info(f'Computing hyperhplane for direction g: {g}')
        g_hat, g_star = compute_hyperplane(mu, nu, g, e_base, emd_kwargs)
        logger.info(f'new half plane: {g}, {g_hat}')
        logger.info(f'new vertex: {g_star}')
        logger.info('Updating bounding boxes')
        P_plus, P_minus = update_box(P_plus, P_minus, [[g, g_hat]], [g_star])
        logger.info(f'Updated box with , P_plus {len(P_plus.V)} P_minus {len(P_minus.V)} vertices.')
        logger.info('Updating c_minus (candidate optimal value)')
        vect_r = g_star@R
        Tcost = 2*np.sum(vect_r[:-1]**2)+vect_r[-1]

        # The optimal values after each iteration is updated
        logger.info('Updating c_plus and c_minus')
        c_plus, x_plus = classical_gw_optimal_cost(np.array(P_plus.V), R)
        logger.info(f'Iteration {iter}: c_minus={c_minus:1.20f}, c_plus={c_plus:1.20f}, c_plus - c_minus = {(c_plus - c_minus):1.20f}, verices in P_minus {len(P_minus.V)} and P_plus {len(P_plus.V)}')
        
        if Tcost > c_minus:
            c_minus = Tcost
            x_minus = g_star
        if c_plus - c_minus < cost_tol:
            break

    # Computing the optimal coupling:
    pi_opt = vect_to_coupling(x_minus, mu, nu, e_base) 
    # Local optimization to finish the optimization 
    c_op, pi_opt, _ = _classical_gw_frank_wolfe(mu, space_x, nu, space_y, pi_opt, g_func, iter_max=FW_iter)

    total_loss = cst_cost-4*c_op
    gap = c_plus - c_op
    lower_bound_on_total_loss = cst_cost-4*c_plus
    logger.info(f'Final results: total loss {total_loss}, gap {gap}, bound on loss {lower_bound_on_total_loss}')

    
    return total_loss, pi_opt, gap, lower_bound_on_total_loss, cst_cost

        
# def gw_m_non_convex(mu, space_x, nu, space_y, emd_kwargs, relax_level=4, cost='IGW', cost_tol=1e-5, iter_max=100, FW_iter=100, t=0.5):
#     space_x, space_y = center_marginal(mu, space_x, nu, space_y,)
#     # Computing constant cost
#     sigma_x = covariance(space_x, mu)
#     sigma_y = covariance(space_y, nu)
#     cst_cost = const_cost(sigma_x, sigma_y, cost, t)
    
#     # Bounding box initialization
#     e_base, R, P_plus, P_minus = initial_box(space_x, space_y, mu, nu, emd_kwargs)
#     R_inv = np.linalg.inv(R)
#     # Initialization of the cost bounds by optimizing non convex function over bounding boxes
#     c_plus, x_plus = optimal_polynomial_cost(P_plus, cost, relax_level, R, t) #need implementation
#     c_minus, x_minus = optimal_polynomial_cost(P_minus, cost, relax_level, R, t)
#     centro = P_minus.get_centroid()
#     init_direc = x_plus - centro
#     init_direc/= np.linalg.norm(init_direc)
#     for iter in range(iter_max):
#         # constraints that will be added to the bounding bow
#         vertex_list = []
#         half_plans_list = []
#         for coup_star, g, g_hat in _frank_wolfe_iter(mu, space_x, nu, space_y, init_direc, R, cost=cost, t=t):
#             # during the iteration, the OT problem (10) is solved with direction g and cost g_hat
#             g_star = projection(coup_star, e_base)
#             # g is in the f basis not in the e basis, and with a wrong format
#             g = f_to_e(g, R_inv)
#             vertex_list.append(g_star)
#             half_plans_list.append([g, g_hat])
#         # The bounding boxes are updated 
#         P_plus, P_minus = update_box(P_plus, P_minus, half_plans_list, vertex_list)
#         # The optimal values after each FW loops are updated
#         c_minus, x_minus = optimal_polynomial_cost(P_minus, cost, relax_level, R, t) 
#         c_plus, x_plus = optimal_polynomial_cost(P_plus, cost, relax_level, R, t)
#         if c_plus - c_minus < cost_tol:
#             break
#         # Choosing next direction 
#         centro = P_minus.get_centroid()
#         init_direc = x_plus - centro
#         init_direc /= np.linalg.norm(init_direc)
#     # Once the algorithm converges, the optimal point is x_minus, we find the appropriate transport plan
#     pi = vect_to_coupling(x_minus, mu, nu, e_base) 
#     # Local optimization 
#     c_op, pi_opt = frank_wolfe_polynomial(mu, space_x, nu, space_y, pi, cost=cost, iter_max = FW_iter, t=t)
    
#     return cst_cost-2*c_op, pi_opt, c_plus - c_minus


def gw_m_non_convex_geometric_approx(mu, space_x, nu, space_y, emd_kwargs, relax_level=2, cost='IGW', geom_tol=1e-8, iter_max=100, FW_iter=100, t=0.5):
    space_x, space_y = center_marginal(mu, space_x, nu, space_y,)
    # Computing constant cost
    sigma_x = covariance(space_x, mu)
    sigma_y = covariance(space_y, nu)
    cst_cost = const_cost(sigma_x, sigma_y, cost, t)
    # Computing bounding box
    P_minus, P_plus, objective, R, e_base = run_approx(mu, nu, space_x, space_y, emd_kwargs, niter=iter_max, epsilon=geom_tol)
    # Global optimization
    logger.info('Starting global polynomial optimization over the approximated polytope')
    _, x_op = optimal_polynomial_cost(P_plus, cost, relax_level, R, t)
    logger.info('Computing optimal coupling from optimal vector')
    pi = vect_to_coupling(x_op, mu, nu, e_base)
    logger.info(f'pi before final FW = {pi}')
    logger.info('Local optimization')
    c_op, pi_opt = frank_wolfe_polynomial(mu, space_x, nu, space_y, pi, cost=cost, iter_max = FW_iter, t=t)
    
    return cst_cost-2*c_op, pi_opt, objective
    
    


def _run_convex_yield(mu, space_x, nu, space_y, emd_kwargs={}, niter = 100, cost='IGW', gap_tol=1e-15, t=None, max_diam=None, convex_tol = 1e-3, p_plus_implementation='cdd'):

    # This code is specifically designed for a convex cost, as IGW or CGW with a high enough t
    d = space_x.shape[-1]
    # selecting the optimale t in the CGW case, if not pre-selected
    t = optimal_t(max_diam, d, cost, t, convex_tol)
    space_x, space_y = center_marginal(mu, space_x, nu, space_y)
    # Computing constant cost
    sigma_x = covariance(space_x, mu)
    sigma_y = covariance(space_y, nu)
    cst_cost = const_cost(sigma_x, sigma_y, cost, t)
    
    # Bounding box initialization
    e_base, R, P_plus, P_minus = initial_box(space_x, space_y, mu, nu, emd_kwargs, p_plus_implementation=p_plus_implementation)
    # print('Initial box with ', len(P_minus.V), ' vertices and ', P_minus.H[0].shape[0], ' half-planes.')
    # print('Vertices: ', P_minus.V)
    
    # Selection of the best direction (lagest score in the bounding box)
    c_plus, x_plus = optimal_cost_cvx(P_plus, cost, R, d, t)
    c_minus, x_minus = optimal_cost_cvx(P_minus, cost, R, d, t)
    
    
    for iter in range(niter):
        
        if c_plus - c_minus < gap_tol:
            break
        # choose direction
        g = new_direction_convex_slow(P_minus, x_plus)
        g_hat, g_star = compute_hyperplane(mu, nu, g, e_base, emd_kwargs)
        P_plus, P_minus = update_box(P_plus, P_minus, [[g, g_hat]], [g_star])
        Tcost = vector_cost(g_star, cost, R, d, t)

        # The optimal values after each iteration is updated
        c_plus, x_plus = optimal_cost_cvx(P_plus, cost, R, d, t)
        logger.info(f'Iteration {iter}: c_minus={c_minus:1.20f}, c_plus={c_plus:1.20f}, c_plus - c_minus = {(c_plus - c_minus):1.20f}, verices in P_minus {len(P_minus.V)} and P_plus {len(P_plus.V)}')
        
        if Tcost > c_minus:
            c_minus = Tcost
            x_minus = g_star
         
        yield P_plus, P_minus, c_plus - c_minus
