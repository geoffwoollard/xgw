import numpy as np
from scipy.spatial.transform import Rotation as R

from xgw.gromov_wasserstein_m_dist import gw_m_convex, gw_m_non_convex, gw_m_non_convex_hausdorff, classical_gw

from test_frank_wolfe import marginals_3d, random_invariance_matrix
from test_hyperplane_approx import make_marginals, make_simple_marginals, marginals

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s.%(msecs)03d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=False,
)

logger = logging.getLogger(__name__)

def testing_2d_classical_gw():
    n_tests = 3
    for test_id in range(n_tests):
        mus, _, space_xs, _ = make_simple_marginals(test_id, d=2)
        
        _, plan, _, _, _ = classical_gw(mus, space_xs, mus, space_xs, {},  cost_tol=1e-15, iter_max=200) 
        plan_error = np.linalg.norm(plan - np.eye(len(mus))/len(mus))
        print('Plan error for identical marginals (classic GW, convex): ', plan_error)
        assert np.isclose(plan_error, 0.0)


def testing_3d_convex():
    '''passing'''

    n_tests = 3
    for test_id in range(n_tests):
        mus, _, space_xs, _ = make_simple_marginals(test_id, d=3, min_points=30, max_points=30)
        r2 = 1
        t = 24*r2 / (2*r2 + 24)
        logger.info(f'Using t={t} for test {test_id}')

        for t_use, cost in zip([None, t*1.01], ['IGW', 'CGW']):
            for p_plus_implementation in ['cdd', 'h_to_v_edges']:    
                logger.info(f'Test {test_id}, cost: {cost}, t_use: {t_use}, p_plus_implementation: {p_plus_implementation}')
                total_loss, pi_opt, gap, lower_bound_on_total_loss, cst_cost = gw_m_convex(mus, space_xs, mus, space_xs, {}, cost=cost, gap_tol=1e-15, iter_max=7, t=t_use, p_plus_implementation=p_plus_implementation) 
                plan_error = np.linalg.norm(pi_opt - np.eye(len(mus))/len(mus))
                logger.info(f'Plan error for identical marginals ({cost}, convex, {p_plus_implementation}): {plan_error}')
                logger.info(f'Total loss: {total_loss:.6f}, lower bound: {lower_bound_on_total_loss:.6f}, gap: {gap:.6f}, constant cost: {cst_cost:.6f}')
                assert np.isclose(plan_error, 0.0), f"Plan error {plan_error} is not close to 0 for identical marginals ({cost}, convex, {p_plus_implementation}) in test {test_id}"
                assert np.isclose(total_loss, 0.0, atol=1e-5), f"Total loss {total_loss} is not close to 0 for identical marginals ({cost}, convex, {p_plus_implementation}) in test {test_id}"


def testing_2d_convex():
    '''passing'''

    n_tests = 3
    for test_id in range(n_tests):
        mus, nus, space_xs, space_ys = make_simple_marginals(test_id, d=2)
        for cost in ['IGW', 'CGW']:
            for p_plus_implementation in ['cdd', 'h_to_v_edges']:
        
                T, plan, c, _, _ = gw_m_convex(mus, space_xs, mus, space_xs, {}, cost=cost, gap_tol=1e-15, iter_max=200, p_plus_implementation=p_plus_implementation) 
                plan_error = np.linalg.norm(plan - np.eye(len(mus))/len(mus))
                print(f'Plan error for identical marginals ({cost}, convex, {p_plus_implementation}): ', plan_error)
                assert np.isclose(plan_error, 0.0)

    mu, nu, space_x, space_y = make_marginals(0)
    near_zero_tolerance = 1e-15
    for cost in ['IGW', 'CGW']:
        for p_plus_implementation in ['cdd', 'h_to_v_edges']:
            final_gap_tolerance = 1e-4
            T, _, gap, _, _ = gw_m_convex(mu, space_x, mu, space_x, {}, cost=cost, gap_tol=near_zero_tolerance, iter_max=200, p_plus_implementation=p_plus_implementation)
            assert -final_gap_tolerance < gap < final_gap_tolerance, f"Gap {gap} is not within tolerance {final_gap_tolerance} for identical marginals ({cost}, convex) in test {test_id} with p_plus_implementation {p_plus_implementation}"
            assert -near_zero_tolerance < T < near_zero_tolerance, f"Total cost {T} is not within tolerance {near_zero_tolerance} for identical marginals ({cost}, convex) in test {test_id} with p_plus_implementation {p_plus_implementation}"

            final_gap_tolerance = 1e-4
            T, _, gap, _, _ = gw_m_convex(mu, space_x, nu, space_y, {}, cost=cost, gap_tol=near_zero_tolerance, iter_max=80, p_plus_implementation=p_plus_implementation) 
            logger.info(f'Test mu, space_x, nu, space_y {cost} convex T: {T}, c: {c}')
            assert -final_gap_tolerance < gap < final_gap_tolerance, f"Gap {gap} is not within tolerance {final_gap_tolerance} for identical marginals ({cost}, convex) in test {test_id} with p_plus_implementation {p_plus_implementation}"
            not_too_small_tolerance = 1e-4
            assert not_too_small_tolerance < T, f"Total cost {T} is too small, should be above {not_too_small_tolerance} for different marginals ({cost}, convex) in test {test_id} with p_plus_implementation {p_plus_implementation}"
    
            gap_tolerance = 5e-3
            T, _, gap, _, _ = gw_m_convex(mus, space_xs, nus, space_ys, {}, cost=cost, gap_tol=near_zero_tolerance, iter_max=200, p_plus_implementation=p_plus_implementation) 
            logger.info(f'Test mus, space_xs, nus, space_ys {cost} convex T: {T}, gap: {gap}, cost: {cost}, p_plus_implementation: {p_plus_implementation}')
            assert -gap_tolerance < gap < gap_tolerance, f"Gap {gap} is not within tolerance {gap_tolerance} for identical marginals ({cost}, convex) in test {test_id} with p_plus_implementation {p_plus_implementation}"
            not_too_small_tolerance = 1e-4
            assert not_too_small_tolerance < T, f"Total cost {T} is too small, should be above {not_too_small_tolerance} for different marginals ({cost}, convex) in test {test_id} with p_plus_implementation {p_plus_implementation}"
    
    
# # def testing_2d_non_convex(marginals):
# #     '''unsure if passing. takes too long to run'''
# #     return 
# #     mu, nu, space_x, space_y = marginals
    
# #     # T, _, c = gw_m_non_convex(mu, space_x, nu, space_y, {}, relax_level=3, cost='DGW', cost_tol=1e-5, iter_max=50, FW_iter = 100)
# #     # assert c < 1e-10
# #     # assert T > 1e-3
    
# #     T, _, c = gw_m_non_convex(mu, space_x, mu, space_x, {}, relax_level=3, cost='DGW', cost_tol=1e-20, iter_max=1000, FW_iter = 100)
# #     assert c < 1e-10
# #     assert T < 1e-10


# # def testing_2d_Hausdorff(marginals):
# #     '''unsure if passing, takes too long to run'''
# #     return 
# #     n_tests = 3
# #     for test_id in range(n_tests):
# #         mus, _, space_xs, _ = make_simple_marginals(test_id, d=2)
    
# #         T, plan, c = gw_m_non_convex_hausdorff(mus, space_xs, mus, space_xs, {}, relax_level=4, cost='DGW', Hausdorff_tol=1e-3, iter_max=20, FW_iter = 100, t=0.5) 
# #         plan_error = np.linalg.norm(plan - np.eye(len(mus))/len(mus))
# #         logger.info(f'Test {test_id}, Hausdorff convex plan error: {plan_error}')
# #         logger.info(f'cost {c}')
# #         logger.info(f'Plan (should be id): {plan}')
# #         assert np.isclose(plan_error, 0.0)
    

def test_igw_convex_reflection_invariant():
    '''passing'''
    n_tests = 3
    for d in [2]:
        for test_id in range(n_tests):
            for p_plus_implementation in ['cdd', 'h_to_v_edges']:
                mus, _, space_xs, _ = make_simple_marginals(test_id, d=d)
                space_xs_reflected = space_xs.copy()
                reflextion_axis = 0
                space_xs_reflected[:,reflextion_axis] = -space_xs_reflected[:,reflextion_axis]

                _, plan, _, _, _ = gw_m_convex(mus, space_xs, mus, space_xs_reflected, {}, cost='IGW', gap_tol=1e-5, iter_max=50, p_plus_implementation=p_plus_implementation)
                mis_match = np.isclose(plan, np.eye(len(mus))/len(mus), atol=1e-7/len(mus)) == False
                logger.info(f'Plan (should be id): {plan}')
                assert np.isclose(mis_match.sum(), 0.0), "Reflection test failed!"


def test_igw_convex_rotation_invariant():
    n_tests = 10
    for d in [2]:
        for test_id in range(n_tests):
            logger.info(f'Test {test_id}, dimension {d}')
            mus, _, space_xs, _ = make_simple_marginals(test_id, d=d, min_points=4, max_points=6)
            np.random.seed(test_id) # ensure same random rotation for each implementation
            random_angle = np.random.rand() * 2 * np.pi
            for p_plus_implementation in ['cdd', 'h_to_v_edges']:
                logger.info(f'Test {test_id}, Implementation: {p_plus_implementation}')
                if d ==2:
                    rotation = R.from_euler('z', random_angle).as_matrix()[:d,:d]
                elif d==3:
                    random_axis = np.random.randn(3)
                    random_axis /= np.linalg.norm(random_axis)
                    rotation = R.from_rotvec(random_axis * random_angle).as_matrix()
                logger.info(f'Rotation matrix: {rotation.shape}')
                space_xs_rotated = space_xs @  rotation.T
                _, plan, _, _, _ = gw_m_convex(mus, space_xs, mus, space_xs_rotated, {}, cost='IGW', gap_tol=1e-5, iter_max=50, p_plus_implementation=p_plus_implementation)
                mis_match = np.isclose(plan, np.eye(len(mus))/len(mus), atol=1e-7/len(mus)) == False
                logger.info(f'Plan (should be id): {plan}')
                assert np.isclose(mis_match.sum(), 0.0), "Rotation test failed!"

def test_cgw_convex_rotation_invariant():
    '''passing'''
    np.random.seed(42)
    n_tests = 3
    for d in [2]:
        for test_id in range(n_tests):
            mus, _, space_xs, _ = make_simple_marginals(test_id, d=d, min_points=10, max_points=20)
            rotation = random_invariance_matrix('CGW', d, random_state=test_id)
            logger.info(f'Rotation matrix: {rotation.shape}')
            space_xs_rotated = space_xs @  rotation.T
            _, plan, _, _, _ = gw_m_convex(mus, space_xs, mus, space_xs_rotated, {}, cost='CGW', gap_tol=1e-5, iter_max=50)
            mis_match = np.isclose(plan, np.eye(len(mus))/len(mus), atol=1e-7/len(mus)) == False
            # mis_match = (plan*len(mus) != np.eye(len(mus)))
            logger.info(f'Plan (should be id): {plan}')
            assert np.isclose(mis_match.sum(), 0.0), "Rotation test failed!"

# # def test_cgw_hausdorff_rotation_invariant_FAILING():
# #     return # currently failing test
# #     np.random.seed(42)
# #     n_tests = 3
# #     for d in [2]:
# #         for test_id in range(n_tests):
# #             mus, _, space_xs, _ = make_simple_marginals(test_id, d=d, min_points=10, max_points=20)
# #             rotation = random_invariance_matrix('CGW', d)
# #             print('Rotation matrix: ', rotation.shape)
# #             space_xs_rotated = space_xs @  rotation.T
# #             _, plan, _ = gw_m_non_convex_Hausdorff(mus, space_xs, mus, space_xs_rotated, {}, relax_level=4, cost='CGW', Hausdorff_tol=1e-5, iter_max=50, FW_iter=100, t=0.7)
# #             mis_match = (plan*len(mus) != np.eye(len(mus)))
# #             print('Plan (should be id): ', plan)
# #             assert np.isclose(mis_match.sum(), 0.0), "Reflection test failed!"


if __name__ == "__main__":
    testing_3d_convex()