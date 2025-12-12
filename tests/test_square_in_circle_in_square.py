import numpy as np
import matplotlib.pyplot as plt
import pytest
import logging
from scipy.spatial import ConvexHull

from xgw.Hyperplane_approx import DoubleRepresentation, Hausdorff, new_direction, update_box, P_plus_outside_P_minus

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@pytest.fixture
def n_iter():
    return 50


def test_Hausdorff():
    inner_square_vertices = [np.array(v) for v in [
        (0,1), (0,-1), (1,0), (-1,0)
    ]]

    outer_square_vertices = [np.array(v) for v in [
        (1,1), (1,-1), (-1,1), (-1,-1)
    ]]

    previous_solutions_to_reuse = {}
    P_plus = DoubleRepresentation()
    P_plus.add_V(outer_square_vertices)  
    P_minus = DoubleRepresentation()
    P_minus.add_V(inner_square_vertices)

    x_0, v_0, objective, previous_solutions_to_reuse = Hausdorff(P_plus, P_minus, previous_solutions_to_reuse)
    
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
    P_plus = DoubleRepresentation()
    P_plus.add_V(outer_square_vertices)  
    P_minus = DoubleRepresentation()
    P_minus.add_V(inner_square_vertices)

    def compute_hyperplane_on_circle_from_direction(direction):
        import numpy as np
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)
        point_on_circle = direction  # since circle of radius 1 centered at origin
        A = direction
        b = np.dot(A, point_on_circle)
        return b, A

    def iteration_loop_circle(P_plus, P_minus, previous_solutions_to_reuse):
        x_0, _, objective, previous_solutions_to_reuse = Hausdorff(P_plus, P_minus, previous_solutions_to_reuse={})
        g = np.random.randn(2)
        g = g / np.linalg.norm(g)
        g_hat, g_star = compute_hyperplane_on_circle_from_direction(g)
        P_plus, P_minus = update_box(P_plus, P_minus, [[g, g_hat]], [g_star])
        return P_plus, P_minus, objective, previous_solutions_to_reuse
    

    np.random.seed(42)
    for _ in range(n_iter):
        P_plus, P_minus, _, previous_solutions_to_reuse = iteration_loop_circle(P_plus=P_plus, P_minus=P_minus, previous_solutions_to_reuse=previous_solutions_to_reuse)
    
        plt.figure()
        list_min = np.array(P_minus.V)
        plt.scatter(list_min[:,0], list_min[:,1], c='k')
        list_pl = np.array(P_plus.V)
        plt.scatter(list_pl[:,0], list_pl[:,1], c='r')
        n_circle = 1000
        theta_grid = np.linspace(0, 2*np.pi,n_circle)
        plt.plot(np.sin(theta_grid),np.cos(theta_grid), c='b')
        plt.title(f'circle approximation iteration n. {iter}')
        plt.savefig(f'img/circle_approx{iter}')
        plt.close()

    residuals = P_plus_outside_P_minus(P_plus, P_minus)
    atol = 1e-15
    assert np.all(residuals <= atol)


    def points_to_volume(points):
        return ConvexHull(points).volume
    
    outer_volume = points_to_volume(np.array(P_plus.V))
    inner_volume = points_to_volume(np.array(P_minus.V))
    assert outer_volume >= inner_volume
    average_volume = (outer_volume + inner_volume) / 2
    assert np.isclose(np.pi, average_volume, atol=0.01)

    plt.clf()
        

def iteration_loop(P_plus, P_minus, previous_solutions_to_reuse):
    x_0, v_0, objective, previous_solutions_to_reuse = Hausdorff(P_plus, P_minus, previous_solutions_to_reuse)
    g = new_direction(x_0, v_0, P_minus)
    g_hat, g_star = 1, g/np.linalg.norm(g)
    P_plus, P_minus = update_box(P_plus, P_minus, [[g, g_hat]], [g_star])
    return P_plus, P_minus, objective, previous_solutions_to_reuse

