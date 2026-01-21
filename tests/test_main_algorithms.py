import numpy as np
from scipy.spatial.transform import Rotation as R

from xgw.Gromov_Wasserstein_m_dist import GW_m_convex, GW_m_non_convex, GW_m_non_convex_Hausdorff

from test_frank_wolfe import marginals_3d
from test_hyperplane_approx import make_marginals, make_simple_marginals, marginals



# def testing_2d_convex():

#     n_tests = 3
#     for test_id in range(n_tests):
#         mus, nus, space_xs, space_ys = make_simple_marginals(test_id, d=2)
        
#         T, plan, c = GW_m_convex(mus, space_xs, mus, space_xs, {}, cost='IGW', cost_tol=1e-15, iter_max=100)
#         plan_error = np.linalg.norm(plan - np.eye(len(mus))/len(mus))
#         print('Plan error for identical marginals (IGW, convex): ', plan_error)
#         assert np.isclose(plan_error, 0.0)

        
#         T, plan, c = GW_m_convex(mus, space_xs, mus, space_xs, {}, cost='CGW', cost_tol=1e-15, iter_max=100, t= 0.67)
#         plan_error = np.linalg.norm(plan - np.eye(len(mus))/len(mus))
#         print('Plan error for identical marginals (CGW, convex): ', plan_error)
#         assert np.isclose(plan_error, 0.0)

#     mu, nu, space_x, space_y = make_marginals(0)

#     T, plan, c = GW_m_convex(mu, space_x, mu, space_x, {}, cost='IGW', cost_tol=1e-15, iter_max=100)
#     plan_error = np.isclose(plan, np.eye(len(mu))/len(mu)).mean()
#     assert plan_error > 0.97

#     T, plan, c = GW_m_convex(mu, space_x, mu, space_x, {}, cost='CGW', cost_tol=1e-15, iter_max=100, t= 0.67)
#     plan_error = np.isclose(plan, np.eye(len(mu))/len(mu)).mean()
#     assert plan_error > 0.97
    
#     cost_tolerance = 0.5

#     T, _, c = GW_m_convex(mu, space_x, nu, space_y, {}, cost='IGW', cost_tol=1e-15, iter_max=200) # need EMD kwarg tuning I guess
#     print(f'Test {test_id} IGW convex T: {T}, c: {c}')
#     assert c < cost_tolerance
#     assert T > 0
    
#     T, _, c = GW_m_convex(mus, space_xs, nus, space_ys, {}, cost='IGW', cost_tol=1e-15, iter_max=200) 
#     print(f'Test {test_id} IGW convex T: {T}, c: {c}')
#     assert c < cost_tolerance
#     assert T > 0
    

#     T, _, c = GW_m_convex(mu, space_x, nu, space_y, {}, cost='CGW', cost_tol=1e-15, iter_max=200, t= 0.67) # need EMD kwarg tuning I guess
#     print(f'Test {test_id} CGW convex T: {T}, c: {c}')
#     assert c < cost_tolerance
#     assert T > 0
    
#     T, _, c = GW_m_convex(mus, space_xs, nus, space_ys, {}, cost='CGW', cost_tol=1e-15, iter_max=200, t= 0.67) 
#     print(f'Test {test_id} CGW convex T: {T}, c: {c}')
#     assert c < cost_tolerance
#     assert T > 0

# def testing_2d_non_convex(marginals):
#     mu, nu, space_x, space_y = marginals
    
#     # T, _, c = GW_m_non_convex(mu, space_x, nu, space_y, {}, relax_level=3, cost='DGW', cost_tol=1e-5, iter_max=50, FW_iter = 100)
#     # assert c<1e-10
#     # assert T>1e-3
    
#     T, _, c = GW_m_non_convex(mu, space_x, mu, space_x, {}, relax_level=3, cost='DGW', cost_tol=1e-20, iter_max=1000, FW_iter = 100)
#     assert c<1e-10
#     assert T<1e-10


# def testing_2d_Hausdorff(marginals):
#     n_tests = 3
#     for test_id in range(n_tests):
#         mus, _, space_xs, _ = make_simple_marginals(test_id, d=2)
    
#         T, plan, c = GW_m_non_convex_Hausdorff(mus, space_xs, mus, space_xs, {}, relax_level=4, cost='DGW', Hausdorff_tol=1e-3, iter_max=20, FW_iter = 100, t=0.5) 
#         plan_error = np.linalg.norm(plan - np.eye(len(mus))/len(mus))
#         print('Test ', test_id, ' Hausdorff convex plan error: ', plan_error)
#         print('cost ', c)
#         print('Plan (should be id): ', plan)
#         assert np.isclose(plan_error, 0.0)
    

# def test_igw_reflection_invariant():
#     n_tests = 3
#     for d in [2,3]:
#         for test_id in range(n_tests):
#             mus, _, space_xs, _ = make_simple_marginals(test_id, d=d)
#             space_xs_reflected = space_xs.copy()
#             reflextion_axis = 0
#             space_xs_reflected[:,reflextion_axis] = -space_xs_reflected[:,reflextion_axis]

#             _, plan, _ = GW_m_convex(mus, space_xs, mus, space_xs_reflected, {}, cost='IGW', cost_tol=1e-5, iter_max=50)
#             mis_match = (plan*len(mus) != np.eye(len(mus)))
#             print('Plan (should be id): ', plan)
#         assert np.isclose(mis_match.sum(), 0.0), "Reflection test failed!"


def test_igw_rotation_invariant():
    n_tests = 3
    for d in [3, 2]:
        for test_id in range(n_tests):
            mus, _, space_xs, _ = make_simple_marginals(test_id, d=d, min_points=4, max_points=6)
            random_angle = np.random.rand() * 2 * np.pi
            if d ==2:
                rotation = R.from_euler('z', random_angle).as_matrix()[:d,:d]
            elif d==3:
                random_axis = np.random.randn(3)
                random_axis /= np.linalg.norm(random_axis)
                rotation = R.from_rotvec(random_axis * random_angle).as_matrix()
            print('Rotation matrix: ', rotation.shape)
            space_xs_rotated = space_xs @  rotation.T
            _, plan, _ = GW_m_convex(mus, space_xs, mus, space_xs_rotated, {}, cost='IGW', cost_tol=1e-5, iter_max=50)
            mis_match = (plan*len(mus) != np.eye(len(mus)))
            print('Plan (should be id): ', plan)
        assert np.isclose(mis_match.sum(), 0.0), "Reflection test failed!"

if __name__ == "__main__":
    test_igw_rotation_invariant()