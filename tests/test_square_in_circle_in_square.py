import os
import numpy as np
import matplotlib.pyplot as plt
import pytest
import logging
from scipy.spatial import ConvexHull

from xgw.hyperplane_approx import DoubleDescription, new_direction, update_box, p_plus_outside_p_minus
from xgw.h_to_v_popcount import masks_from_B
from xgw.hyperplane_approx import build_B_from_H_and_V

from xgw.utils import canonicalize_vertices, canonicalize_halfspaces

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

@pytest.fixture
def p_minus_dual_implementations():
    return ['h_to_v_popcount', 'h_to_v_popcount_sparse']

def compute_hyperplane_on_circle_from_direction(direction):
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)
    point_on_circle = direction  # since circle of radius 1 centered at origin
    A = direction
    b = np.dot(A, point_on_circle)
    return b, A
    
def test_H_and_V_after_cut(p_plus_implementations, p_minus_implementations, p_minus_dual_implementations):
    inner_square_vertices = [np.array(v) for v in [
        (0,1), (0,-1), (1,0), (-1,0)
    ]]

    outer_square_vertices = [np.array(v) for v in [
        (1,1), (1,-1), (-1,1), (-1,-1)
    ]]

    one_over_sqrt_2 = 1/np.sqrt(2)
    p_minus_vertices_true = np.array([
        [-1.     ,  0.     ],
        [-one_over_sqrt_2, -one_over_sqrt_2],
        [ 0.     , -1.     ],
        [ 0.     ,  1.     ],
        [ 1.     ,  0.     ]])
    p_minus_vertices_true = canonicalize_vertices(p_minus_vertices_true)

    p_plus_vertices_true = np.array([
        [-1.     , 1 - 2*one_over_sqrt_2],
        [-1.     ,  1.     ],
        [1 - 2*one_over_sqrt_2, -1.     ],
        [ 1.     , -1.     ],
        [ 1.     ,  1.     ]])
    p_plus_vertices_true = canonicalize_vertices(p_plus_vertices_true)

    p_plus_A_true = np.array([
        [-0.70710678, -0.70710678],
        [-0.        , -1.        ],
        [-1.        , -0.        ],
        [-0.        ,  1.        ],
        [ 1.        , -0.        ]])
    p_plus_b_true = np.ones(len(p_plus_A_true))
    p_plus_A_true, p_plus_b_true = canonicalize_halfspaces(p_plus_A_true, p_plus_b_true)

    p_minus_A_true = np.array([
        [-one_over_sqrt_2,  one_over_sqrt_2],
        [-0.38268343, -0.92387953],
        [-0.92387953, -0.38268343],
        [ one_over_sqrt_2, -one_over_sqrt_2],
        [ one_over_sqrt_2,  one_over_sqrt_2]])
    p_minus_b_true = np.array([0.70710678, 0.92387953, 0.92387953, 0.70710678, 0.70710678])
    p_minus_A_true, p_minus_b_true = canonicalize_halfspaces(p_minus_A_true, p_minus_b_true)

    g = -one_over_sqrt_2*np.ones(2)

    for p_plus_implementation in p_plus_implementations:
        B_initialization = None
        masks_initialization = None
        if p_plus_implementation == 'cdd':
            p_plus = DoubleDescription(implementation=p_plus_implementation)
            p_plus.add_V(outer_square_vertices)  
        elif p_plus_implementation == 'h_to_v_edges':
            p_plus = DoubleDescription(implementation=p_plus_implementation, E_initialization=np.array([[0,1],[0,2],[1,3],[2,3]]))
            p_plus.V = outer_square_vertices
            A = np.array([[1,0],[0,1],[-1,0],[0,-1]])
            b = np.array([1,1,1,1])
            p_plus.H = [A, b]
        elif p_plus_implementation == 'h_to_v_popcount':
            B_initialization = np.array([
                [1,1,0,0],
                [1,0,1,0],
                [0,0,1,1],
                [0,1,0,1],
            ], dtype=bool) 
            A = np.array([[1,0],[0,1],[-1,0],[0,-1]])
            b = np.array([1,1,1,1])
            p_plus = DoubleDescription(implementation=p_plus_implementation, V_initialization=np.array(outer_square_vertices), B_initialization=B_initialization, A_initialization=A, b_initialization=b)
            p_plus.V = outer_square_vertices
            p_plus.H = [A, b]
        elif p_plus_implementation == 'h_to_v_popcount_sparse':
            B_initialization = np.array([
                [1,1,0,0],
                [1,0,1,0],
                [0,0,1,1],
                [0,1,0,1],
            ], dtype=bool) 
            masks_initialization = masks_from_B(B_initialization)
            B_initialization = None
            A = np.array([[1,0],[0,1],[-1,0],[0,-1]])
            b = np.array([1,1,1,1])
            p_plus = DoubleDescription(implementation=p_plus_implementation, V_initialization=np.array(outer_square_vertices), B_initialization=B_initialization, masks_initialization=masks_initialization, A_initialization=A, b_initialization=b)
            p_plus.V = outer_square_vertices
            p_plus.H = [A, b]
        else:
            raise ValueError(f'Unknown p_plus_implementation {p_plus_implementation}')
        p_minus_init = DoubleDescription()
        p_minus_init.add_V(inner_square_vertices)


        for p_minus_implementation in p_minus_implementations:
            p_minus_dual_implementations = ['cdd','h_to_v_popcount', 'h_to_v_popcount_sparse'] if p_minus_implementation == 'v_to_h_dual' else [None]
            for p_minus_dual_implementation in p_minus_dual_implementations:
                p_minus = DoubleDescription(implementation=p_minus_implementation, 
                                            dual_implementation=p_minus_dual_implementation, 
                                            V_initialization=np.array(p_minus_init.V),
                                            A_initialization=p_minus_init.H[0], 
                                            b_initialization=p_minus_init.H[1])
                for v in inner_square_vertices:
                    p_minus.add_V([v])


                g_hat, g_star = compute_hyperplane_on_circle_from_direction(g)
                p_plus, p_minus = update_box(p_plus, p_minus, [[g, g_hat]], [g_star])


                p_minus_vertices_canonical = canonicalize_vertices(p_minus.V)
                p_plus_vertices_canonical = canonicalize_vertices(p_plus.V)
                assert np.allclose(p_minus_vertices_canonical, p_minus_vertices_true), f'fail for p_plus_implementation {p_plus_implementation}, p_minus vertices: {p_minus_vertices_canonical}'
                assert np.allclose(p_plus_vertices_canonical, p_plus_vertices_true), f'fail for p_plus_implementation {p_plus_implementation}, p_plus vertices: {p_plus_vertices_canonical}'

                p_minus_A_canonical, p_minus_b_canonical = canonicalize_halfspaces(p_minus.H[0], p_minus.H[1])
                p_plus_A_canonical, p_plus_b_canonical = canonicalize_halfspaces(p_plus.H[0], p_plus.H[1])
                assert np.allclose(p_minus_A_canonical, p_minus_A_true), f'fail for p_plus_implementation {p_plus_implementation}, p_minus A: {p_minus_A_canonical}, expected: {p_minus_A_true}'
                assert np.allclose(p_minus_b_canonical, p_minus_b_true), f'fail for p_plus_implementation {p_plus_implementation}, p_minus b: {p_minus_b_canonical}, expected: {p_minus_b_true}'
                assert np.allclose(p_plus_A_canonical, p_plus_A_true), f'fail for p_plus_implementation {p_plus_implementation}, p_plus A: {p_plus_A_canonical}, expected: {p_plus_A_true}'
                assert np.allclose(p_plus_b_canonical, p_plus_b_true), f'fail for p_plus_implementation {p_plus_implementation}, p_plus b: {p_plus_b_canonical}, expected: {p_plus_b_true}'

    print("test_H_and_V_after_cut passed")

def test_pi_2dim(n_iter, p_plus_implementations, p_minus_implementations, p_minus_dual_implementations):
    inner_square_vertices = [np.array(v) for v in [
        (0,1), (0,-1), (1,0), (-1,0)
    ]]

    outer_square_vertices = [np.array(v) for v in [
        (1,1), (1,-1), (-1,1), (-1,-1) # 01, 02, 13, 23
    ]]

    for p_plus_implementation in p_plus_implementations:
        for p_minus_implementation in p_minus_implementations:
            dual_implementations = p_minus_dual_implementations if p_minus_implementation == 'v_to_h_dual' else [None]
            for p_minus_dual_implementation in dual_implementations:
                for p_plus_use_D_sparse in [True, False] if p_minus_dual_implementation in ['h_to_v_popcount', 'h_to_v_popcount_sparse'] else [None]:
                    for p_minus_use_D_sparse_dual in [True, False] if p_minus_dual_implementation in ['h_to_v_popcount', 'h_to_v_popcount_sparse'] else [None]:
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
                            p_plus = DoubleDescription(implementation=p_plus_implementation, V_initialization=np.array(outer_square_vertices), B_initialization=B_initialization, use_D_sparse=p_plus_use_D_sparse)
                            p_plus.V = outer_square_vertices
                        elif p_plus_implementation == 'h_to_v_popcount_sparse':
                            B_initialization = np.array([
                                [1,1,0,0],
                                [1,0,1,0],
                                [0,0,1,1],
                                [0,1,0,1],
                            ], dtype=np.uint8)
                            masks_initialization = masks_from_B(B_initialization)
                            p_plus = DoubleDescription(implementation=p_plus_implementation, V_initialization=np.array(outer_square_vertices), masks_initialization=masks_initialization, A_initialization=A, b_initialization=b, use_D_sparse_dual=p_minus_use_D_sparse_dual)
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
                                                    b_initialization=b_p_minus,
                                                    use_D_sparse_dual=p_minus_use_D_sparse_dual)
                        for v in inner_square_vertices:
                            print(f"Adding vertex {v} to p_minus via dual with primal implementation {p_minus_implementation} and dual implementation {p_minus_dual_implementation}...")
                            p_minus.add_V([v])



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
                            if not os.path.exists('tests/results/test_square_in_circle_in_square'): # mkdir if not exists
                                os.makedirs('tests/results/test_square_in_circle_in_square')
                            plt.savefig(f'tests/results/test_square_in_circle_in_square/circle_approx_pplusimplementation_{p_plus_implementation.replace("_", "")}__pplususeDsparsedual{p_plus_use_D_sparse}_pminusimplementation_{p_minus_implementation.replace("_", "")}_pminusdualimplementation_{str(p_minus_dual_implementation).replace("_", "")}_pminususeDsparsedual{p_minus_use_D_sparse_dual}_iter_{iter}.png', dpi=200)
                            plt.close()

                        residuals = p_plus_outside_p_minus(p_plus, p_minus)
                        atol = 1e-6
                        assert np.all(residuals <= atol), f"Some points in p_plus are outside p_minus by more than {atol} for implementation {p_plus_implementation}, residuals: {np.max(residuals)}"

                        outer_volume, inner_volume, average_volume = area_estimate(p_plus, p_minus)
                        assert outer_volume >= inner_volume
                        assert np.isclose(np.pi, average_volume, atol=0.01), f'average volume {average_volume} is not close to pi, p_plus implementation {p_plus_implementation}, p_minus implementation {p_minus_implementation}, p_minus dual implementation {p_minus_dual_implementation}'

                        plt.clf()


def make_inner_square_vertices(n_dim):
    """Generate inner square (cross/diamond) vertices in n dimensions.
    Vertices at ±1 along each axis: (±1, 0, 0, ...), (0, ±1, 0, ...), etc.
    Total: 2*n_dim vertices.
    """
    vertices = []
    for i in range(n_dim):
        v_pos = np.zeros(n_dim)
        v_pos[i] = 1
        vertices.append(v_pos)
        
        v_neg = np.zeros(n_dim)
        v_neg[i] = -1
        vertices.append(v_neg)
    
    return [np.array(v) for v in vertices]

def make_outer_square_vertices(n_dim):
    """Generate outer square (hypercube) vertices in n dimensions.
    All ±1 corner combinations: (±1, ±1, ..., ±1).
    Total: 2^n_dim vertices.
    """
    from itertools import product
    vertices = list(product([-1, 1], repeat=n_dim))
    return [np.array(v, dtype=float) for v in vertices]

def volume_monte_carlo(A, b, n_samples=100000):
    """Estimate convex hull volume via Monte Carlo."""
    d = A.shape[1]
    
    # find bounding box
    # from scipy.optimize import linprog
    # bounds = []
    # for i in range(d):
    #     c = np.zeros(d)
    #     c[i] = 1
    #     res_min = linprog(c, A_ub=A, b_ub=b, bounds=None, method='highs')
    #     res_max = linprog(-c, A_ub=A, b_ub=b, bounds=None, method='highs')
    #     bounds.append((res_min.x[i], res_max.x[i]))
    lower = -np.ones(d)
    upper = np.ones(d)
    
    # sample uniformly in bbox
    # lower = np.array([lb for lb, _ in bounds])
    # upper = np.array([ub for _, ub in bounds])
    
    samples = np.random.uniform(
        lower,
        upper,
        size=(n_samples, d)
    )
    
    # check feasibility: A @ x <= b
    feasible = np.all(A @ samples.T <= b[:, None], axis=0)
    
    # volume estimate
    bbox_volume = np.prod(upper - lower)
    est_volume = bbox_volume * np.mean(feasible)
    
    return est_volume

def volume_estimate_fast(p_plus, p_minus):
    """Fast volume estimate using Monte Carlo sampling."""
    from scipy.optimize import linprog
    
    A_plus, b_plus = np.array(p_plus.H[0]), np.array(p_plus.H[1])
    A_minus, b_minus = np.array(p_minus.H[0]), np.array(p_minus.H[1])
    
    # Monte Carlo for outer volume
    outer_volume = volume_monte_carlo(A_plus, b_plus, n_samples=100000)
    inner_volume = volume_monte_carlo(A_minus, b_minus, n_samples=100000)
    
    average_volume = (outer_volume + inner_volume) / 2
    return outer_volume, inner_volume, average_volume

def test_vol_ndim(n_iter, p_plus_implementations, p_minus_implementations, p_minus_dual_implementations, n_dim=3):
    inner_square_vertices = make_inner_square_vertices(n_dim)
    outer_square_vertices = make_outer_square_vertices(n_dim)

    n_dim_volume = np.pi**(n_dim/2) / np.math.gamma(n_dim/2 + 1)

    for p_plus_implementation in p_plus_implementations:
        for p_minus_implementation in p_minus_implementations:
            dual_implementations = p_minus_dual_implementations if p_minus_implementation == 'v_to_h_dual' else [None]
            for p_minus_dual_implementation in dual_implementations:
                for p_plus_use_D_sparse in [True, ] if p_minus_dual_implementation in ['h_to_v_popcount', 'h_to_v_popcount_sparse'] else [None]:
                    for p_minus_use_D_sparse_dual in [True, ] if p_minus_dual_implementation in ['h_to_v_popcount', 'h_to_v_popcount_sparse'] else [None]:
                        print(f'Testing implementation {p_plus_implementation} with p_minus implementation {p_minus_implementation}...')
                        p_plus_init = DoubleDescription(implementation='cdd')
                        p_plus_init.add_V(outer_square_vertices) 
                        A, b = p_plus_init.H 
                        B_initialization = build_B_from_H_and_V(A, b, np.array(outer_square_vertices))
                        if p_plus_implementation == 'cdd':
                            p_plus = p_plus_init
                        elif p_plus_implementation == 'h_to_v_edges':
                            from xgw.h_to_v_edges import find_edges
                            E_initialization = find_edges( np.array(outer_square_vertices))
                            p_plus = DoubleDescription(implementation=p_plus_implementation, E_initialization=E_initialization)
                            p_plus.V = outer_square_vertices
                        elif p_plus_implementation == 'h_to_v_popcount':
                            p_plus = DoubleDescription(implementation=p_plus_implementation, V_initialization=np.array(outer_square_vertices), B_initialization=B_initialization, use_D_sparse=p_plus_use_D_sparse)
                            p_plus.V = outer_square_vertices
                        elif p_plus_implementation == 'h_to_v_popcount_sparse':
                            masks_initialization = masks_from_B(B_initialization)
                            p_plus = DoubleDescription(implementation=p_plus_implementation, V_initialization=np.array(outer_square_vertices), masks_initialization=masks_initialization, A_initialization=A, b_initialization=b, use_D_sparse_dual=p_minus_use_D_sparse_dual)
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
                                                    b_initialization=b_p_minus,
                                                    use_D_sparse_dual=p_minus_use_D_sparse_dual)
                        for v in inner_square_vertices:
                            print(f"Adding vertex {v} to p_minus via dual with primal implementation {p_minus_implementation} and dual implementation {p_minus_dual_implementation}...")
                            p_minus.add_V([v])

                        def iteration_loop_circle(p_plus, p_minus):
                            g, _ = new_direction(p_plus, p_minus, tol=1e-10)
                            g_hat, g_star = compute_hyperplane_on_circle_from_direction(g)
                            p_plus, p_minus = update_box(p_plus, p_minus, [[g, g_hat]], [g_star])
                            return p_plus, p_minus
                    
                        def points_to_volume(points):
                            if points.shape[1] > 4:
                                print(f"Skipping convex hull volume computation for {points.shape[1]}-dimensional points due to computational complexity.")
                                return np.nan
                            try:
                                vol = ConvexHull(points).volume
                            except Exception as e1:
                                try:
                                    print(f"ConvexHull failed for points: {points}, error: {e1}. Adding jitter and retrying...")
                                    jitter = np.random.normal(scale=1e-10, size=points.shape)
                                    vol = ConvexHull(points + jitter).volume
                                except Exception as e2:
                                    print(f"Failed to compute volume for points: {points}, error: {e2}")
                                    vol = np.nan
                            return vol
                        
                        def volume_estimate(p_plus, p_minus):
                            outer_volume = points_to_volume(np.array(p_plus.V))
                            inner_volume = points_to_volume(np.array(p_minus.V))
                            average_volume = (outer_volume + inner_volume) / 2
                            return outer_volume, inner_volume, average_volume    

                        np.random.seed(42)
                        for iter in range(n_iter):
                            print('iteration', iter)
                            p_plus, p_minus = iteration_loop_circle(p_plus, p_minus)
                            print(f'p_plus vertices: {len(p_plus.V)}, p_plus halfspaces: {p_plus.H[0].shape[0]}, p_minus vertices: {len(p_minus.V)}, p_minus halfspaces: {p_minus.H[0].shape[0]}')
                            for vol_func, func_name in zip([volume_estimate, volume_estimate_fast], ['convex hull volume estimate', 'monte carlo volume estimate']):
                                outer_volume, inner_volume, average_volume = vol_func(p_plus, p_minus)
                                rel_error = abs(average_volume - n_dim_volume) / n_dim_volume
                                print(f'iteration {iter}: inner volume {inner_volume:<10.5f}, outer volume {outer_volume:<10.5f}, average volume {average_volume:<10.5f}, target volume {n_dim_volume:<10.5f}, rel error {rel_error:<10.5f}. Volume estimate method: {func_name}')

                        residuals = p_plus_outside_p_minus(p_plus, p_minus)
                        atol = 1e-6
                        assert np.all(residuals <= atol), f"Some points in p_plus are outside p_minus by more than {atol} for implementation {p_plus_implementation}, residuals: {np.max(residuals)}"

                        outer_volume, inner_volume, average_volume = volume_estimate_fast(p_plus, p_minus)
                        assert outer_volume >= inner_volume
                        rel_error = abs(average_volume - n_dim_volume) / n_dim_volume
                        assert np.isclose(rel_error, 0.05, atol=0.1), f'average volume {average_volume} is not close enough to {n_dim_volume}, p_plus implementation {p_plus_implementation}, p_minus implementation {p_minus_implementation}, p_minus dual implementation {p_minus_dual_implementation}'

                        plt.clf()

if __name__ == "__main__":
    # test_H_and_V_after_cut(['cdd'], ['cdd', 'v_to_h_dual'])
    test_pi_ndim(n_iter=50, p_plus_implementations=['h_to_v_popcount',], p_minus_implementations=['v_to_h_dual'], p_minus_dual_implementations=['h_to_v_popcount'], n_dim=2)
