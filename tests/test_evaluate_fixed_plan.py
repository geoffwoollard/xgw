import numpy as np

from xgw.gromov_wasserstein_m_dist import gw_m_convex
from xgw.evalulate_fixed_plan import evaluate_cost

from test_hyperplane_approx import make_marginals, make_simple_marginals

def test_evaluate_cost():
    t = 8 / (2 + 8)
    n_trials = 3
    for cost in ['IGW', 'CGW']:
        for seed in range(n_trials):
            mus, nus, space_xs, space_ys = make_marginals(seed, min_points=10, max_points=10)
            r_max = max(np.linalg.norm(space_xs, axis=1).max(), np.linalg.norm(space_ys, axis=1).max())
            space_xs /= r_max
            space_ys /= r_max
            total_loss, plan, _, _, _ = gw_m_convex(mus, space_xs, nus, space_ys, {}, cost=cost, gap_tol=1e-5, iter_max=50, t=t)
            total_loss_direct_eval = evaluate_cost(space_xs, space_ys, plan, cost, t)
            assert np.isclose(total_loss, total_loss_direct_eval, atol=1e-6), f"Cost from GW: {total_loss}, Cost from evaluation: {total_loss_direct_eval}, seed: {seed}, cost: {cost}"

            mus, nus, space_xs, space_ys = make_simple_marginals(seed, d=2, min_points=10, max_points=10)
            r_max = max(np.linalg.norm(space_xs, axis=1).max(), np.linalg.norm(space_ys, axis=1).max())
            space_xs /= r_max
            space_ys /= r_max
            total_loss, plan, _, _, _ = gw_m_convex(mus, space_xs, nus, space_ys, {}, cost=cost, gap_tol=1e-5, iter_max=50, t=t)
            total_loss_direct_eval = evaluate_cost(space_xs, space_ys, plan, cost, t)
            assert np.isclose(total_loss, total_loss_direct_eval, atol=1e-6), f"Cost from GW: {total_loss}, Cost from evaluation: {total_loss_direct_eval}, seed: {seed}, cost: {cost}"

if __name__ == "__main__":
    test_evaluate_cost()