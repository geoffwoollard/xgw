import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R

from xgw.gromov_wasserstein_m_dist import gw_m_convex, classical_gw, gw_m_non_convex_geometric_approx

from test_frank_wolfe import marginals_3d, random_invariance_matrix
from test_hyperplane_approx import make_marginals, make_simple_marginals, marginals, make_marginals_preturbed

import logging

# logging.basicConfig(
#     level=logging.WARNING,
#     format="%(asctime)s.%(msecs)03d - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
#     force=False,
# )

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def test_2d_classical_gw():
    '''passing'''
    return
    n_tests = 3
    for test_id in range(n_tests):
        mus, _, space_xs, _ = make_simple_marginals(test_id, d=2)
        
        _, plan, _, _, _ = classical_gw(mus, space_xs, mus, space_xs, {},  cost_tol=1e-8, iter_max=50) 
        plan_error = np.linalg.norm(plan - np.eye(len(mus))/len(mus))
        logger.info(f'Plan error for identical marginals (classic GW, convex):  {plan_error}') 
        assert np.isclose(plan_error, 0.0)
        
        M = random_invariance_matrix('IGW', 2, random_state=test_id)
        _, plan, _, _, _ = classical_gw(mus, space_xs, mus, space_xs@M, {},  cost_tol=1e-8, iter_max=50) 
        plan_error = np.linalg.norm(plan - np.eye(len(mus))/len(mus))
        logger.info(f'Plan error for identical marginals (classic GW, convex):  {plan_error}')
        assert np.isclose(plan_error, 0.0)

@pytest.fixture
def p_plus_implementations_to_test():
    return ['cdd', 'h_to_v_edges', 'h_to_v_popcount', 'h_to_v_popcount_sparse'] 

@pytest.fixture
def p_minus_implementations_to_test():
    return ['cdd', 'v_to_h_dual']

@pytest.fixture
def p_minus_dual_implementations_to_test():
    return ['h_to_v_popcount','h_to_v_popcount_sparse', ]

def test_3d_convex(p_plus_implementations_to_test, p_minus_implementations_to_test, p_minus_dual_implementations_to_test, iter_max=10):
    '''passing'''
    n_tests = 3
    for test_id in range(n_tests):
        mus, nus, space_xs, space_ys = make_simple_marginals(test_id, d=3, min_points=30, max_points=30)
        r2 = 1
        t = 8*r2 / (2*r2 + 8)
        logger.info(f'Using t={t} for test {test_id}')

        for t_use, cost in zip([None, t*1.01], ['IGW', 'CGW']):
            for p_plus_implementation in p_plus_implementations_to_test:
                for p_minus_implementation in p_minus_implementations_to_test:
                    for p_minus_dual_implementation in p_minus_dual_implementations_to_test if p_minus_implementation == 'v_to_h_dual' else [None]:
                        logger.info(f'Test {test_id}, cost: {cost}, t_use: {t_use}, p_plus_implementation: {p_plus_implementation}')
                        total_loss, pi_opt, gap, lower_bound_on_total_loss, cst_cost = gw_m_convex(mus, space_xs, mus, space_xs, {}, cost=cost, gap_tol=1e-15, iter_max=iter_max, t=t_use, p_plus_implementation=p_plus_implementation, FW_iter=500, p_minus_implementation=p_minus_implementation, p_minus_dual_implementation=p_minus_dual_implementation) 
                        plan_error = np.linalg.norm(pi_opt - np.eye(len(mus))/len(mus))
                        logger.info(f'Plan error for identical marginals ({cost}, convex, {p_plus_implementation}): {plan_error}')
                        logger.info(f'Total loss: {total_loss:.6f}, lower bound: {lower_bound_on_total_loss:.6f}, gap: {gap:.6f}, constant cost: {cst_cost:.6f}')
                        assert np.isclose(plan_error, 0.0), f"Plan error {plan_error} is not close to 0 for identical marginals ({cost}, convex, {p_plus_implementation}) in test {test_id}"
                        assert np.isclose(total_loss, 0.0, atol=1e-5), f"Total loss {total_loss} is not close to 0 for identical marginals ({cost}, convex, {p_plus_implementation}) in test {test_id}"

                        logger.info(f'Test {test_id}, cost: {cost}, t_use: {t_use}, p_plus_implementation: {p_plus_implementation} - different simple marginals')
                        total_loss, pi_opt, gap, lower_bound_on_total_loss, cst_cost = gw_m_convex(mus, space_xs, nus, space_ys, {}, cost=cost, gap_tol=1e-15, iter_max=iter_max, t=t_use, p_plus_implementation=p_plus_implementation, FW_iter=500, p_minus_implementation=p_minus_implementation, p_minus_dual_implementation=p_minus_dual_implementation) 
                        logger.info(f'Total loss: {total_loss:.6f}, lower bound: {lower_bound_on_total_loss:.6f}, gap: {gap:.6f}, constant cost: {cst_cost:.6f}')

def test_2d_convex(p_plus_implementations_to_test, p_minus_implementations_to_test, p_minus_dual_implementations_to_test):
    '''passing'''
    return
    n_tests = 3
    for test_id in range(n_tests):
        mus, nus, space_xs, space_ys = make_simple_marginals(test_id, d=2)
        for cost in ['IGW', 'CGW']:
            for p_plus_implementation in p_plus_implementations_to_test:
                for p_minus_implementation in p_minus_implementations_to_test:
                    for p_minus_dual_implementation in p_minus_dual_implementations_to_test if p_minus_implementation == 'v_to_h_dual' else [None]:
                        print(f'Test {test_id}, cost: {cost}, p_plus_implementation: {p_plus_implementation}')
                        T, plan, c, _, _ = gw_m_convex(mus, space_xs, mus, space_xs, {}, cost=cost, gap_tol=1e-15, iter_max=200, p_plus_implementation=p_plus_implementation, p_minus_implementation=p_minus_implementation, p_minus_dual_implementation=p_minus_dual_implementation) 
                        plan_error = np.linalg.norm(plan - np.eye(len(mus))/len(mus))
                        print(f'Plan error for identical marginals ({cost}, convex, {p_plus_implementation}): ', plan_error)
                        assert np.isclose(plan_error, 0.0), f"Plan error {plan_error} is not close to 0 for identical marginals ({cost}, convex, {p_plus_implementation}) in test {test_id}"

    
    mu, nu, space_x, space_y = make_marginals(0)
    near_zero_tolerance = 1e-15
    for cost in ['IGW', 'CGW']:
        for p_plus_implementation in p_plus_implementations_to_test: #TODO: add in h_to_v_edges
            print(f'Test {test_id}, cost: {cost}, p_plus_implementation: {p_plus_implementation}')

            final_gap_tolerance_pos = 1e-4
            final_gap_tolerance_neg = -0.01
            print(f'Test {test_id}, cost: {cost}, p_plus_implementation: {p_plus_implementation} - identical simple marginals')
            T, _, gap, _, _ = gw_m_convex(mu, space_x, mu, space_x, {}, cost=cost, gap_tol=near_zero_tolerance, iter_max=200, FW_iter=200, p_plus_implementation=p_plus_implementation)
            assert final_gap_tolerance_neg < gap < final_gap_tolerance_pos, f"Gap {gap} is not within tolerance {(final_gap_tolerance_neg, final_gap_tolerance_pos)} for identical marginals ({cost}, convex) in test {test_id} with p_plus_implementation {p_plus_implementation}"
            assert -near_zero_tolerance < T < near_zero_tolerance, f"Total cost {T} is not within tolerance {near_zero_tolerance} for identical marginals ({cost}, convex) in test {test_id} with p_plus_implementation {p_plus_implementation}"

            final_gap_tolerance_pos = 1e-4
            final_gap_tolerance_neg = -0.01
            print(f'Test {test_id}, cost: {cost}, p_plus_implementation: {p_plus_implementation} - different simple marginals')
            T, _, gap, _, _ = gw_m_convex(mu, space_x, nu, space_y, {}, cost=cost, gap_tol=near_zero_tolerance, iter_max=80, p_plus_implementation=p_plus_implementation) 
            print(f'Test mu, space_x, nu, space_y {cost} convex T: {T}, gap: {gap}, cost: {cost}, p_plus_implementation: {p_plus_implementation}')
            assert final_gap_tolerance_neg < gap < final_gap_tolerance_pos, f"Gap {gap} is not within tolerance {(final_gap_tolerance_neg, final_gap_tolerance_pos)} for identical marginals ({cost}, convex) in test {test_id} with p_plus_implementation {p_plus_implementation}"
            not_too_small_tolerance = 1e-4
            assert not_too_small_tolerance < T, f"Total cost {T} is too small, should be above {not_too_small_tolerance} for different marginals ({cost}, convex) in test {test_id} with p_plus_implementation {p_plus_implementation}"
    
            final_gap_tolerance_pos = 5e-3
            final_gap_tolerance_neg = -0.01
            print(f'Test {test_id}, cost: {cost}, p_plus_implementation: {p_plus_implementation} - multiple marginals')
            T, _, gap, _, _ = gw_m_convex(mus, space_xs, nus, space_ys, {}, cost=cost, gap_tol=near_zero_tolerance, iter_max=200, p_plus_implementation=p_plus_implementation) 
            logger.info(f'Test mus, space_xs, nus, space_ys {cost} convex T: {T}, gap: {gap}, cost: {cost}, p_plus_implementation: {p_plus_implementation}')
            assert final_gap_tolerance_neg < gap < final_gap_tolerance_pos, f"Gap {gap} is not within tolerance {(final_gap_tolerance_neg, final_gap_tolerance_pos)} for identical marginals ({cost}, convex) in test {test_id} with p_plus_implementation {p_plus_implementation}"
            not_too_small_tolerance = 1e-4
            assert not_too_small_tolerance < T, f"Total cost {T} is too small, should be above {not_too_small_tolerance} for different marginals ({cost}, convex) in test {test_id} with p_plus_implementation {p_plus_implementation}"
    
def test_2d_non_convex(marginals):
    '''passing'''

    mu, nu, space_x, space_y = marginals
    
    T, _, c = gw_m_non_convex_geometric_approx(mu, space_x, nu, space_y, {}, relax_level=2, cost='DGW', geom_tol=1e-15, iter_max=300, FW_iter = 200)
    assert c < 5e-4, f"c={c}"
    assert T > 1e-3, f"T={T}"
    
    T, plan, c = gw_m_non_convex_geometric_approx(mu, space_x, mu, space_x, {}, relax_level=2, cost='DGW', geom_tol=1e-15, iter_max=200, FW_iter = 200)
    assert c < 6e-4, f"c={c}"
    assert T < 1e-10, f"T={T}"
    plan_error = np.linalg.norm(plan - np.diag(mu))
    logger.info(f'Non-convex plan error: {plan_error}')
    logger.info(f'cost {c}')
    logger.info(f'Plan (should be id): {plan}')
    assert np.isclose(plan_error, 0.0)
    
    # checking invariance under SL transform
    transform = random_invariance_matrix('DGW', 2)
    space_x_transformed = space_x @  transform.T
    T, plan, c = gw_m_non_convex_geometric_approx(mu, space_x, mu, space_x_transformed, {}, relax_level=2, cost='DGW', geom_tol=1e-15, iter_max=200, FW_iter = 200)
    assert c < 6e-4, f"c={c}"
    assert T < 1e-1, f"T={T}" # todo: should be smaller. passing on clement's local env 
    plan_error = np.linalg.norm(plan - np.diag(mu))
    logger.info(f'Non-convex plan error: {plan_error}')
    logger.info(f'cost {c}')
    logger.info(f'Plan (should be id): {plan}')
    assert np.isclose(plan_error, 0.0)

def test_igw_convex_reflection_invariant(p_plus_implementations_to_test):
    '''passing'''
    return
    n_tests = 3
    for d in [2]:
        for test_id in range(n_tests):
            for p_plus_implementation in p_plus_implementations_to_test:
                mus, _, space_xs, _ = make_simple_marginals(test_id, d=d)
                space_xs_reflected = space_xs.copy()
                reflextion_axis = 0
                space_xs_reflected[:,reflextion_axis] = -space_xs_reflected[:,reflextion_axis]
                cost = 'IGW' 
                T, plan, gap, _, _ = gw_m_convex(mus, space_xs, mus, space_xs_reflected, {}, cost=cost, gap_tol=1e-5, iter_max=50, p_plus_implementation=p_plus_implementation)
                logger.info(f'Test mus, space_xs, mus, space_xs_reflected {cost} convex T: {T}, gap: {gap}, cost: {cost}, p_plus_implementation: {p_plus_implementation}')
                mis_match = np.isclose(plan, np.eye(len(mus))/len(mus), atol=1e-7/len(mus)) == False
                logger.info(f'Plan (should be id): {plan}')
                assert np.isclose(mis_match.sum(), 0.0), "Reflection test failed!"
                
                
                space_xs_transformed = space_xs@random_invariance_matrix('IGW', d, random_state=test_id)
                _, plan, _, _, _ = gw_m_convex(mus, space_xs, mus, space_xs_transformed, {}, cost='IGW', gap_tol=1e-5, iter_max=50, p_plus_implementation=p_plus_implementation)
                mis_match = np.isclose(plan, np.eye(len(mus))/len(mus), atol=1e-7/len(mus)) == False
                logger.info(f'Plan (should be id): {plan}')
                assert np.isclose(mis_match.sum(), 0.0), "Reflection test failed!"
                
                
                space_xs_transformed = space_xs@random_invariance_matrix('IGW', d, random_state=test_id)
                _, plan, _, _, _ = gw_m_convex(mus, space_xs, mus, space_xs_transformed, {}, cost='IGW', gap_tol=1e-5, iter_max=50, p_plus_implementation=p_plus_implementation)
                mis_match = np.isclose(plan, np.eye(len(mus))/len(mus), atol=1e-7/len(mus)) == False
                logger.info(f'Plan (should be id): {plan}')
                assert np.isclose(mis_match.sum(), 0.0), "Reflection test failed!"

def test_igw_convex_rotation_invariant(p_plus_implementations_to_test):
    return
    n_tests = 10
    for d in [2]:
        for test_id in range(n_tests):
            logger.info(f'Test {test_id}, dimension {d}')
            mus, _, space_xs, _ = make_simple_marginals(test_id, d=d, min_points=4, max_points=6)
            np.random.seed(test_id) # ensure same random rotation for each implementation
            for p_plus_implementation in p_plus_implementations_to_test:
                logger.info(f'Test {test_id}, Implementation: {p_plus_implementation}')
                rotation = random_invariance_matrix('CGW', d, random_state=test_id)
                logger.info(f'Rotation matrix: {rotation.shape}')
                space_xs_rotated = space_xs @  rotation.T
                cost = 'IGW'
                T, plan, gap, _, _ = gw_m_convex(mus, space_xs, mus, space_xs_rotated, {}, cost=cost, gap_tol=1e-5, iter_max=50, p_plus_implementation=p_plus_implementation)
                logger.info(f'Test mus, space_xs, mus, space_xs_rotated {cost} convex T: {T}, gap: {gap}, cost: {cost}, p_plus_implementation: {p_plus_implementation}')
                mis_match = np.isclose(plan, np.eye(len(mus))/len(mus), atol=1e-7/len(mus)) == False
                assert np.isclose(mis_match.sum(), 0.0), f"Rotation test failed! Plan (should be id): {plan}"

def test_cgw_convex_rotation_invariant():
    '''passing'''
    return
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

            logger.info(f'Plan (should be id): {plan}')
            assert np.isclose(mis_match.sum(), 0.0), "Rotation test failed!"




if __name__ == "__main__":
    test_2d_non_convex(make_marginals_preturbed(0, 0.01, 0))
