import os
import numpy as np
import matplotlib.pyplot as plt
import pytest
import logging
from scipy.spatial import ConvexHull

from xgw.hyperplane_approx import DoubleDescription, new_direction, update_box, p_plus_outside_p_minus
from xgw.h_to_v_popcount import masks_from_B

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@pytest.fixture
def n_iter():
    return 25

@pytest.fixture
def p_plus_implementations():
    return ['cdd', 'h_to_v_edges', 'h_to_v_popcount', 'h_to_v_popcount_sparse']

@pytest.fixture
def p_minus_implementations():
    return ['cdd', 'v_to_h_dual']

def test_hausdorff(p_plus_implementations):
    inner_square_vertices = [np.array(v) for v in [
        (0,1), (0,-1), (1,0), (-1,0)
    ]]

    outer_square_vertices = [np.array(v) for v in [
        (1,1), (1,-1), (-1,1), (-1,-1)
    ]]

    for implementation in p_plus_implementations:
        previous_solutions_to_reuse = {}
        B_initialization = None
        masks_initialization = None
        if implementation == 'cdd':
            p_plus = DoubleDescription(implementation=implementation)
            p_plus.add_V(outer_square_vertices)  
        elif implementation == 'h_to_v_edges':
            p_plus = DoubleDescription(implementation=implementation, E_initialization=np.array([[0,1],[1,2],[2,3],[3,0]]))
            p_plus.V = outer_square_vertices
            A = np.array([[1,0],[0,1],[-1,0],[0,-1]])
            b = np.array([1,1,1,1])
            p_plus.H = [A, b]
        elif implementation == 'h_to_v_popcount':
            B_initialization = np.array([
                [1,1,0,0],
                [1,0,1,0],
                [0,0,1,1],
                [0,1,0,1],
            ], dtype=np.uint8)
        elif implementation == 'h_to_v_popcount_sparse':
            B_initialization = np.array([
                [1,1,0,0],
                [1,0,1,0],
                [0,0,1,1],
                [0,1,0,1],
            ], dtype=np.uint8)
            masks_initialization = masks_from_B(B_initialization)
            B_initialization = None
            A = np.array([[1,0],[0,1],[-1,0],[0,-1]])
            b = np.array([1,1,1,1])
            p_plus = DoubleDescription(implementation=implementation, V_initialization=np.array(outer_square_vertices), B_initialization=B_initialization, masks_initialization=masks_initialization, A_initialization=A, b_initialization=b)
            p_plus.V = outer_square_vertices
            p_plus.H = [A, b]
        else:
            raise ValueError(f'Unknown implementation {implementation}')
        p_minus = DoubleDescription()
        p_minus.add_V(inner_square_vertices)

#         x_0, v_0, objective, previous_solutions_to_reuse = hausdorff(p_plus, p_minus, previous_solutions_to_reuse)
        
#         assert np.allclose(objective, np.linalg.norm(x_0 - v_0)**2), f'fail for implementation {implementation}'
#         assert np.allclose(2*x_0, v_0), f'fail for implementation {implementation}' # since the closest point in the inner square to a vertex of the outer square is at half the distance


def test_minimal_2d(n_iter, p_plus_implementations, p_minus_implementations):
    inner_square_vertices = [np.array(v) for v in [
        (0,1), (0,-1), (1,0), (-1,0)
    ]]

    outer_square_vertices = [np.array(v) for v in [
        (1,1), (1,-1), (-1,1), (-1,-1) # 01, 02, 13, 23
    ]]

    for p_plus_implementation in p_plus_implementations:
        for p_minus_implementation in p_minus_implementations:
            dual_implementations = ['cdd', 'h_to_v_popcount'] if p_minus_implementation == 'v_to_h_dual' else [None]
            for p_minus_dual_implementation in dual_implementations:
                print(f'Testing implementation {p_plus_implementation} with p_minus implementation {p_minus_implementation}...')
                A = np.array([[1,0],[0,1],[-1,0],[0,-1]])
                b = np.array([1,1,1,1])

                if p_plus_implementation == 'cdd':
                    p_plus = DoubleDescription(implementation=p_plus_implementation)
                    p_plus.add_V(outer_square_vertices)  
                elif p_plus_implementation == 'h_to_v_edges':
                    p_plus = DoubleDescription(implementation=p_plus_implementation, E_initialization=np.array([[0,1],[0,2],[1,3],[2,3]]))
                    p_plus.V = outer_square_vertices
                elif p_plus_implementation == 'h_to_v_popcount':
                    B_initialization = np.array([
                        [1,1,0,0],
                        [1,0,1,0],
                        [0,0,1,1],
                        [0,1,0,1],
                    ], dtype=np.uint8)
                    p_plus = DoubleDescription(implementation=p_plus_implementation, V_initialization=np.array(outer_square_vertices), B_initialization=B_initialization)
                    p_plus.V = outer_square_vertices
                elif p_plus_implementation == 'h_to_v_popcount_sparse':
                    B_initialization = np.array([
                        [1,1,0,0],
                        [1,0,1,0],
                        [0,0,1,1],
                        [0,1,0,1],
                    ], dtype=np.uint8)
                    masks_initialization = masks_from_B(B_initialization)
                    p_plus = DoubleDescription(implementation=p_plus_implementation, V_initialization=np.array(outer_square_vertices), masks_initialization=masks_initialization, A_initialization=A, b_initialization=b)
                    p_plus.V = outer_square_vertices
                else:
                    raise ValueError(f'Unknown implementation {p_plus_implementation}')

                p_plus.H = [A, b]
                p_minus_base = DoubleDescription()
                p_minus_base.add_V(inner_square_vertices)
                A_p_minus, b_p_minus = p_minus_base.H
                p_minus = DoubleDescription(implementation=p_minus_implementation, 
                                            dual_implementation=p_minus_dual_implementation, 
                                            V_initialization=np.array(inner_square_vertices),
                                            A_initialization=A_p_minus, 
                                            b_initialization=b_p_minus)
                for v in inner_square_vertices:
                    print(f"Adding vertex {v} to p_minus via dual with primal implementation {p_minus_implementation} and dual implementation {p_minus_dual_implementation}...")
                    p_minus.add_V([v])

                def compute_hyperplane_on_circle_from_direction(direction):
                    direction = np.array(direction)
                    direction = direction / np.linalg.norm(direction)
                    point_on_circle = direction  # since circle of radius 1 centered at origin
                    A = direction
                    b = np.dot(A, point_on_circle)
                    return b, A

                def iteration_loop_circle(p_plus, p_minus):
                    g, _ = new_direction(p_plus, p_minus, tol=1e-10)
                    g_hat, g_star = compute_hyperplane_on_circle_from_direction(g)
                    p_plus, p_minus = update_box(p_plus, p_minus, [[g, g_hat]], [g_star])
                    return p_plus, p_minus
            
                def points_to_volume(points):
                    return ConvexHull(points).volume
                
                def area_estimate(p_plus, p_minus):
                    outer_volume = points_to_volume(np.array(p_plus.V))
                    inner_volume = points_to_volume(np.array(p_minus.V))
                    average_volume = (outer_volume + inner_volume) / 2
                    return outer_volume, inner_volume, average_volume    

                np.random.seed(42)
                for iter in range(n_iter):
                    print('iteration', iter)
                    p_plus, p_minus = iteration_loop_circle(p_plus, p_minus)
                    outer_volume, inner_volume, average_volume = area_estimate(p_plus, p_minus)
                    print(f'iteration {iter}: inner volume {inner_volume}, outer volume {outer_volume}, average volume {average_volume}')

                    # print('number of vertices p_plus', len(p_plus.V))
                    # print('number of half-planes p_plus', len(p_plus.H[0]))
                    # print('vertices:', p_plus.V)
                    # print('half-planes:', p_plus.H)
                
                    plt.figure()
                    list_min = np.array(p_minus.V)
                    plt.scatter(list_min[:,0], list_min[:,1], c='k',  label=r'$P_{\Pi}^-$')
                    list_pl = np.array(p_plus.V)
                    plt.scatter(list_pl[:,0], list_pl[:,1], c='r', label=r'$P_{\Pi}^+$')
                    n_circle = 1000
                    theta_grid = np.linspace(0, 2*np.pi,n_circle)
                    plt.plot(np.sin(theta_grid),np.cos(theta_grid), c='b')
                    plt.title(f'circle approximation iteration n. {iter}')
                    plt.legend()
                    if not os.path.exists('tests/results'): # mkdir if not exists
                        os.makedirs('tests/results')
                    plt.savefig(f'tests/results/circle_approx_pplusimplementation_{p_plus_implementation.replace("_", "")}_pminusimplementation_{p_minus_implementation.replace("_", "")}_pminusdualimplementation_{str(p_minus_dual_implementation).replace("_", "")}_iter_{iter}.png', dpi=200)
                    plt.close()

                residuals = p_plus_outside_p_minus(p_plus, p_minus)
                atol = 1e-6
                assert np.all(residuals <= atol), f"Some points in p_plus are outside p_minus by more than {atol} for implementation {p_plus_implementation}, residuals: {np.max(residuals)}"

                outer_volume, inner_volume, average_volume = area_estimate(p_plus, p_minus)
                assert outer_volume >= inner_volume
                assert np.isclose(np.pi, average_volume, atol=0.01), f'average volume {average_volume} is not close to pi, p_plus implementation {p_plus_implementation}, p_minus implementation {p_minus_implementation}, p_minus dual implementation {p_minus_dual_implementation}'

                plt.clf()
            
if __name__ == "__main__":
    test_minimal_2d(30, ['cdd'], ['v_to_h_dual'])