import os
import numpy as np
import matplotlib.pyplot as plt
import pytest
import logging
from scipy.spatial import ConvexHull

from xgw.hyperplane_approx import DoubleDescription, hausdorff, new_direction, update_box, p_plus_outside_p_minus

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@pytest.fixture
def n_iter():
    return 50


def test_hausdorff():
    inner_square_vertices = [np.array(v) for v in [
        (0,1), (0,-1), (1,0), (-1,0)
    ]]

    outer_square_vertices = [np.array(v) for v in [
        (1,1), (1,-1), (-1,1), (-1,-1)
    ]]

    previous_solutions_to_reuse = {}
    p_plus = DoubleDescription()
    p_plus.add_V(outer_square_vertices)  
    p_minus = DoubleDescription()
    p_minus.add_V(inner_square_vertices)

    x_0, v_0, objective, previous_solutions_to_reuse = hausdorff(p_plus, p_minus, previous_solutions_to_reuse)
    
    assert np.allclose(objective, np.linalg.norm(x_0 - v_0)**2)
    assert np.allclose(2*x_0, v_0) # since the closest point in the inner square to a vertex of the outer square is at half the distance


def test_minimal_2d(n_iter):
    inner_square_vertices = [np.array(v) for v in [
        (0,1), (0,-1), (1,0), (-1,0)
    ]]

    outer_square_vertices = [np.array(v) for v in [
        (1,1), (1,-1), (-1,1), (-1,-1)
    ]]

    previous_solutions_to_reuse = {}
    p_plus = DoubleDescription()
    p_plus.add_V(outer_square_vertices)  
    p_minus = DoubleDescription()
    p_minus.add_V(inner_square_vertices)

    def compute_hyperplane_on_circle_from_direction(direction):
        import numpy as np
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)
        point_on_circle = direction  # since circle of radius 1 centered at origin
        A = direction
        b = np.dot(A, point_on_circle)
        return b, A

    def iteration_loop_circle(p_plus, p_minus, previous_solutions_to_reuse):
        x_0, _, objective, previous_solutions_to_reuse = hausdorff(p_plus, p_minus, previous_solutions_to_reuse={})
        g = np.random.randn(2)
        g = g / np.linalg.norm(g)
        g_hat, g_star = compute_hyperplane_on_circle_from_direction(g)
        p_plus, p_minus = update_box(p_plus, p_minus, [[g, g_hat]], [g_star])
        return p_plus, p_minus, objective, previous_solutions_to_reuse
    

    np.random.seed(42)
    for iter in range(n_iter):
        p_plus, p_minus, _, previous_solutions_to_reuse = iteration_loop_circle(p_plus=p_plus, p_minus=p_minus, previous_solutions_to_reuse=previous_solutions_to_reuse)
    
        plt.figure()
        list_min = np.array(p_minus.V)
        plt.scatter(list_min[:,0], list_min[:,1], c='k',  label=r'$P_{\Pi}^=$')
        list_pl = np.array(p_plus.V)
        plt.scatter(list_pl[:,0], list_pl[:,1], c='r', label=r'$P_{\Pi}^+$')
        n_circle = 1000
        theta_grid = np.linspace(0, 2*np.pi,n_circle)
        plt.plot(np.sin(theta_grid),np.cos(theta_grid), c='b')
        plt.title(f'circle approximation iteration n. {iter}')
        plt.legend()
        # mkdir if not exists
        if not os.path.exists('tests/results'):
            os.makedirs('tests/results')
        plt.savefig(f'tests/results/circle_approx{iter}', dpi=50)
        plt.close()

    residuals = p_plus_outside_p_minus(p_plus, p_minus)
    atol = 1e-15
    assert np.all(residuals <= atol)


    def points_to_volume(points):
        return ConvexHull(points).volume
    
    outer_volume = points_to_volume(np.array(p_plus.V))
    inner_volume = points_to_volume(np.array(p_minus.V))
    assert outer_volume >= inner_volume
    average_volume = (outer_volume + inner_volume) / 2
    assert np.isclose(np.pi, average_volume, atol=0.01)

    plt.clf()
        

def iteration_loop(p_plus, p_minus, previous_solutions_to_reuse):
    x_0, v_0, objective, previous_solutions_to_reuse = hausdorff(p_plus, p_minus, previous_solutions_to_reuse)
    g = new_direction(x_0, v_0, p_minus)
    g_hat, g_star = 1, g/np.linalg.norm(g)
    p_plus, p_minus = update_box(p_plus, p_minus, [[g, g_hat]], [g_star])
    return p_plus, p_minus, objective, previous_solutions_to_reuse

