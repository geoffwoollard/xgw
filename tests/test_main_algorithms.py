import numpy as np
from xgw.Gromov_Wasserstein_m_dist import GW_m_convex, GW_m_non_convex, GW_m_non_convex_Hausdorff
from test_frank_wolfe import marginals_3d
from test_hyperplane_approx import marginals, simple_marginals_2D



def testing_2d_convex(marginals, simple_marginals_2D):
    mu, nu, space_x, space_y = marginals
    mus, nus, space_xs, space_ys = simple_marginals_2D
    
    # T, _, c = GW_m_convex(mu, space_x, mu, space_x, {}, cost='IGW', cost_tol=1e-15, iter_max=1000)
    # assert c<1e-15 and c>=-1e-15
    # assert T<1e-15 and T>=-1e-15
    
    # T, _, c = GW_m_convex(mu, space_x, mu, space_x, {}, cost='CGW', cost_tol=1e-20, iter_max=1000, t= 0.67)
    # assert c<1e-15 and c>=-1e-15
    # assert T<1e-15 and T>=-1e-15
    
    
    # T, _, c = GW_m_convex(mu, space_x, nu, space_y, {}, cost='IGW', cost_tol=1e-15, iter_max=1000) # need EMD kwarg tuning I guess
    # assert c<1e-5
    # assert T>1e-3
    
    # T, _, c = GW_m_convex(mus, space_xs, nus, space_ys, {}, cost='IGW', cost_tol=1e-15, iter_max=1000) 
    # assert c<1e-10
    # assert T>1e-3
    

    T, _, c = GW_m_convex(mu, space_x, nu, space_y, {}, cost='CGW', cost_tol=1e-15, iter_max=1000, t= 0.67) # need EMD kwarg tuning I guess
    assert c<1e-5
    assert T>1e-3
    
    # T, _, c = GW_m_convex(mus, space_xs, nus, space_ys, {}, cost='CGW', cost_tol=1e-15, iter_max=1000, t= 0.67) 
    # assert c<1e-10
    # assert T>1e-3
    

# def testing_2d_non_convex(marginals):
#     mu, nu, space_x, space_y = marginals
    
#     # T, _, c = GW_m_non_convex(mu, space_x, nu, space_y, {}, relax_level=3, cost='DGW', cost_tol=1e-5, iter_max=50, FW_iter = 100)
#     # assert c<1e-10
#     # assert T>1e-3
    
#     T, _, c = GW_m_non_convex(mu, space_x, mu, space_x, {}, relax_level=3, cost='DGW', cost_tol=1e-20, iter_max=1000, FW_iter = 100)
#     assert c<1e-10
#     assert T<1e-10


# def testing_2d_Hausdorff(marginals):
#     mu, nu, space_x, space_y = marginals
    
#     T, _, c = GW_m_non_convex_Hausdorff(mu, space_x, nu, space_y, {}, relax_level=4, cost='DGW', Hausdorff_tol=1e-8, iter_max=100, FW_iter = 100, t=0.5)
#     assert c<1e-8
#     assert T>1e-3
    
#     T, _, c = GW_m_non_convex_Hausdorff(mu, space_x, nu, space_y, {}, relax_level=4, cost='DGW', Hausdorff_tol=1e-8, iter_max=100, FW_iter = 100, t=0.5)
#     assert c<1e-8
#     assert T<1e-10
