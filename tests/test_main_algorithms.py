import numpy as np
from scipy.spatial.transform import Rotation as R

from xgw.gromov_wasserstein_m_dist import gw_m_convex, gw_m_non_convex, gw_m_non_convex_hausdorff

from test_frank_wolfe import marginals_3d, random_invariance_matrix
from test_hyperplane_approx import make_marginals, make_simple_marginals, marginals

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

logger = logging.getLogger(__name__)


def testing_2d_convex():
    '''passing'''

    n_tests = 1
    for test_id in range(n_tests):
        mus, nus, space_xs, space_ys = make_simple_marginals(test_id, d=2)
        
        T, plan, c = gw_m_convex(mus, space_xs, mus, space_xs, {}, cost='IGW', cost_tol=1e-15, iter_max=200) # imax 200 crashes
        plan_error = np.linalg.norm(plan - np.eye(len(mus))/len(mus))
        print('Plan error for identical marginals (IGW, convex): ', plan_error)
        assert np.isclose(plan_error, 0.0)

        
        T, plan, c = gw_m_convex(mus, space_xs, mus, space_xs, {}, cost='CGW', cost_tol=1e-15, iter_max=50)
        plan_error = np.linalg.norm(plan - np.eye(len(mus))/len(mus))
        print('Plan error for identical marginals (CGW, convex): ', plan_error)
        assert np.isclose(plan_error, 0.0)

    mu, nu, space_x, space_y = make_marginals(0)
    cost_tolerance = 1e-5 # NB: fails if 1e-6

    T, plan, c = gw_m_convex(mu, space_x, mu, space_x, {}, cost='IGW', cost_tol=1e-15, iter_max=500)
    assert -cost_tolerance < c < cost_tolerance 
    assert -1e-15 < T < 1e-15

    T, plan, c = gw_m_convex(mu, space_x, mu, space_x, {}, cost='CGW', cost_tol=1e-15, iter_max=500)
    assert -cost_tolerance < c < cost_tolerance 
    assert -1e-15 < T < 1e-15
    
    cost_tolerance = 1e-4
    T, _, c = gw_m_convex(mu, space_x, nu, space_y, {}, cost='IGW', cost_tol=1e-15, iter_max=50) 
    logger.info(f'Test 1 IGW convex T: {T}, c: {c}')
    assert -cost_tolerance < c < cost_tolerance 
    not_too_small_tolerance = 1e-2
    assert not_too_small_tolerance < T
    
    cost_tolerance = 5e-3
    T, _, c = gw_m_convex(mus, space_xs, nus, space_ys, {}, cost='IGW', cost_tol=1e-15, iter_max=200) 
    logger.info(f'Test 2 IGW convex T: {T}, c: {c}')
    assert -cost_tolerance < c < cost_tolerance 
    not_too_small_tolerance = 1
    assert not_too_small_tolerance < T
    
    cost_tolerance = 1e-3
    T, _, c = gw_m_convex(mu, space_x, nu, space_y, {}, cost='CGW', cost_tol=1e-15, iter_max=50) # need EMD kwarg tuning I guess
    logger.info(f'Test {test_id} IGW convex T: {T}, c: {c} 3')
    assert -cost_tolerance < c < cost_tolerance 
    not_too_small_tolerance = 1e-2
    assert not_too_small_tolerance < T
    
    cost_tolerance = 5e-3
    T, _, c = gw_m_convex(mus, space_xs, nus, space_ys, {}, cost='CGW', cost_tol=1e-15, iter_max=200) 
    logger.info(f'Test {test_id} IGW convex T: {T}, c: {c} 4')
    assert -cost_tolerance < c < cost_tolerance 
    not_too_small_tolerance = 1
    assert not_too_small_tolerance < T
    
# def testing_2d_non_convex(marginals):
#     '''unsure if passing. takes too long to run'''
#     return 
#     mu, nu, space_x, space_y = marginals
    
#     # T, _, c = gw_m_non_convex(mu, space_x, nu, space_y, {}, relax_level=3, cost='DGW', cost_tol=1e-5, iter_max=50, FW_iter = 100)
#     # assert c < 1e-10
#     # assert T > 1e-3
    
#     T, _, c = gw_m_non_convex(mu, space_x, mu, space_x, {}, relax_level=3, cost='DGW', cost_tol=1e-20, iter_max=1000, FW_iter = 100)
#     assert c < 1e-10
#     assert T < 1e-10


# def testing_2d_Hausdorff(marginals):
#     '''unsure if passing, takes too long to run'''
#     return 
#     n_tests = 3
#     for test_id in range(n_tests):
#         mus, _, space_xs, _ = make_simple_marginals(test_id, d=2)
    
#         T, plan, c = gw_m_non_convex_hausdorff(mus, space_xs, mus, space_xs, {}, relax_level=4, cost='DGW', Hausdorff_tol=1e-3, iter_max=20, FW_iter = 100, t=0.5) 
#         plan_error = np.linalg.norm(plan - np.eye(len(mus))/len(mus))
#         logger.info(f'Test {test_id}, Hausdorff convex plan error: {plan_error}')
#         logger.info(f'cost {c}')
#         logger.info(f'Plan (should be id): {plan}')
#         assert np.isclose(plan_error, 0.0)
    

def test_igw_convex_reflection_invariant():
    '''passing'''
    n_tests = 3
    for d in [2]:
        for test_id in range(n_tests):
            mus, _, space_xs, _ = make_simple_marginals(test_id, d=d)
            space_xs_reflected = space_xs.copy()
            reflextion_axis = 0
            space_xs_reflected[:,reflextion_axis] = -space_xs_reflected[:,reflextion_axis]

            _, plan, _ = gw_m_convex(mus, space_xs, mus, space_xs_reflected, {}, cost='IGW', cost_tol=1e-5, iter_max=50)
            mis_match = (plan*len(mus) != np.eye(len(mus)))
            logger.info(f'Plan (should be id): {plan}')
            assert np.isclose(mis_match.sum(), 0.0), "Reflection test failed!"


# def test_igw_convex_rotation_invariant():
#     '''sometimes failing (depending random seed), because of vertex outside (internal assert statement)
    
#             def new_direction_convex_slow(P_minus, x_plus):
#             # Normalized normal vectors of P_minus
#             A,b =  P_minus.H
#             # Finding best direction
#             testing_dir = A @ x_plus - b
#             index = np.argmax(testing_dir)
#             g = A[index]
#             # checking that the point is outside P_minus
#     >       assert g @ x_plus - b[index]>=0

#     '''
#     np.random.seed(0)
#     n_tests = 1
#     for d in [2]:
#         for test_id in range(n_tests):
#             mus, _, space_xs, _ = make_simple_marginals(test_id, d=d, min_points=4, max_points=6)
#             random_angle = np.random.rand() * 2 * np.pi
#             if d ==2:
#                 rotation = R.from_euler('z', random_angle).as_matrix()[:d,:d]
#             elif d==3:
#                 random_axis = np.random.randn(3)
#                 random_axis /= np.linalg.norm(random_axis)
#                 rotation = R.from_rotvec(random_axis * random_angle).as_matrix()
#             print('Rotation matrix: ', rotation.shape)
#             space_xs_rotated = space_xs @  rotation.T
#             _, plan, _ = gw_m_convex(mus, space_xs, mus, space_xs_rotated, {}, cost='IGW', cost_tol=1e-5, iter_max=50)
#             mis_match = (plan*len(mus) != np.eye(len(mus)))
#             print('Plan (should be id): ', plan)
#             assert np.isclose(mis_match.sum(), 0.0), "Reflection test failed!"

def test_cgw_convex_rotation_invariant():
    '''passing'''
    np.random.seed(42)
    n_tests = 3
    for d in [2]:
        for test_id in range(n_tests):
            mus, _, space_xs, _ = make_simple_marginals(test_id, d=d, min_points=10, max_points=20)
            rotation = random_invariance_matrix('CGW', d)
            print('Rotation matrix: ', rotation.shape)
            space_xs_rotated = space_xs @  rotation.T
            _, plan, _ = gw_m_convex(mus, space_xs, mus, space_xs_rotated, {}, cost='CGW', cost_tol=1e-5, iter_max=50)
            mis_match = (plan*len(mus) != np.eye(len(mus)))
            print('Plan (should be id): ', plan)
            assert np.isclose(mis_match.sum(), 0.0), "Reflection test failed!"

# def test_cgw_hausdorff_rotation_invariant_FAILING():
#     return # currently failing test
#     np.random.seed(42)
#     n_tests = 3
#     for d in [2]:
#         for test_id in range(n_tests):
#             mus, _, space_xs, _ = make_simple_marginals(test_id, d=d, min_points=10, max_points=20)
#             rotation = random_invariance_matrix('CGW', d)
#             print('Rotation matrix: ', rotation.shape)
#             space_xs_rotated = space_xs @  rotation.T
#             _, plan, _ = gw_m_non_convex_Hausdorff(mus, space_xs, mus, space_xs_rotated, {}, relax_level=4, cost='CGW', Hausdorff_tol=1e-5, iter_max=50, FW_iter=100, t=0.7)
#             mis_match = (plan*len(mus) != np.eye(len(mus)))
#             print('Plan (should be id): ', plan)
#             assert np.isclose(mis_match.sum(), 0.0), "Reflection test failed!"


if __name__ == "__main__":
    testing_2d_convex()