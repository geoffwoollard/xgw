import numpy as np
from xgw.Gromov_Wasserstein_m_dist import GW_m_convex, GW_m_non_convex, GW_m_non_convex_Hausdorff
from test_frank_wolfe import marginals_3d
from test_hyperplane_approx import marginals



def testing_2d(marginals):
    mu, nu, space_x, space_y = marginals
    
    T, _, c = GW_m_convex(mu, space_x, mu, space_x, {}, cost='IGW', cost_tol=1e-15, iter_max=100)
    assert c<1e-10
    assert T<1e-10
    
    T, _, c = GW_m_convex(mu, space_x, mu, space_x, {}, cost='CGW', cost_tol=1e-15, iter_max=1000, t= 0.6)
    assert c<1e-4
    assert T<1e-4
    
    # Works but need to optimize the different solver parameters
    
