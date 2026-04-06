import os
import numpy as np
import matplotlib.pyplot as plt
import pytest
import logging
from scipy.spatial import ConvexHull
from copy import deepcopy

from xgw.hyperplane_approx import DoubleDescription, new_direction, update_box, p_plus_outside_p_minus
from xgw.h_to_v_popcount import masks_from_B
from xgw.hyperplane_approx import build_B_from_H_and_V

from xgw.utils import canonicalize_vertices, canonicalize_halfspaces

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@pytest.fixture
def n_iter():
    return 25

# @pytest.fixture
def p_plus_implementations():
    return ['cdd', 'h_to_v_edges', 'h_to_v_popcount', 'h_to_v_popcount_sparse']

# @pytest.fixture
def p_minus_implementations():
    return ['cdd', 'v_to_h_dual']

# @pytest.fixture
def p_minus_dual_implementations():
    return ['cdd','h_to_v_popcount', 'h_to_v_popcount_sparse']

def compute_hyperplane_on_sphere_from_direction(direction):
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)
    point_on_circle = direction  # since circle of radius 1 centered at origin
    A = direction
    b = np.dot(A, point_on_circle)
    return b, A

def iteration_loop_sphere(p_plus, p_minus):
    g, _ = new_direction(p_plus, p_minus, tol=1e-10)
    g_hat, g_star = compute_hyperplane_on_sphere_from_direction(g)
    p_plus, p_minus = update_box(p_plus, p_minus, [[g, g_hat]], [g_star])
    return p_plus, p_minus
    
def points_to_volume_convex_hull(points):
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

def volume_estimate_convex_hull(p_plus, p_minus):
    outer_volume = points_to_volume_convex_hull(np.array(p_plus.V))
    inner_volume = points_to_volume_convex_hull(np.array(p_minus.V))
    average_volume = (outer_volume + inner_volume) / 2
    return outer_volume, inner_volume, average_volume   
    
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
            dual_implementations = p_minus_dual_implementations if p_minus_implementation == 'v_to_h_dual' else [None]
            for p_minus_dual_implementation in dual_implementations:
                p_plus_fresh = deepcopy(p_plus)
                print(f"Testing p_plus implementation {p_plus_implementation} with p_minus implementation {p_minus_implementation} and dual implementation {p_minus_dual_implementation}...")
                p_minus = DoubleDescription(implementation=p_minus_implementation, 
                                            dual_implementation=p_minus_dual_implementation, 
                                            V_initialization=np.array(p_minus_init.V),
                                            A_initialization=p_minus_init.H[0], 
                                            b_initialization=p_minus_init.H[1])
                for v in inner_square_vertices:
                    p_minus.add_V([v])


                g_hat, g_star = compute_hyperplane_on_sphere_from_direction(g)
                p_plus_fresh, p_minus = update_box(p_plus_fresh, p_minus, [[g, g_hat]], [g_star])


                p_minus_vertices_canonical = canonicalize_vertices(p_minus.V)
                p_plus_vertices_canonical = canonicalize_vertices(p_plus_fresh.V)
                assert np.allclose(p_minus_vertices_canonical, p_minus_vertices_true), f'fail for p_plus_implementation {p_plus_implementation}, p_minus vertices: {p_minus_vertices_canonical}'
                assert np.allclose(p_plus_vertices_canonical, p_plus_vertices_true), f'fail for p_plus_implementation {p_plus_implementation}, p_plus vertices: {p_plus_vertices_canonical}'

                p_minus_A_canonical, p_minus_b_canonical = canonicalize_halfspaces(p_minus.H[0], p_minus.H[1])
                p_plus_A_canonical, p_plus_b_canonical = canonicalize_halfspaces(p_plus_fresh.H[0], p_plus_fresh.H[1])
                assert np.allclose(p_minus_A_canonical, p_minus_A_true), f'fail for p_plus_implementation {p_plus_implementation}, p_minus A: {p_minus_A_canonical}, expected: {p_minus_A_true}'
                assert np.allclose(p_minus_b_canonical, p_minus_b_true), f'fail for p_plus_implementation {p_plus_implementation}, p_minus b: {p_minus_b_canonical}, expected: {p_minus_b_true}'
                assert np.allclose(p_plus_A_canonical, p_plus_A_true), f'fail for p_plus_implementation {p_plus_implementation}, p_plus A: {p_plus_A_canonical}, expected: {p_plus_A_true}'
                assert np.allclose(p_plus_b_canonical, p_plus_b_true), f'fail for p_plus_implementation {p_plus_implementation}, p_plus b: {p_plus_b_canonical}, expected: {p_plus_b_true}'

    print("test_H_and_V_after_cut passed")

def test_square_in_circle_in_square(n_iter, p_plus_implementations, p_minus_implementations, p_minus_dual_implementations):
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
                            ], dtype=bool)
                            p_plus = DoubleDescription(implementation=p_plus_implementation, V_initialization=np.array(outer_square_vertices), B_initialization=B_initialization, use_D_sparse=p_plus_use_D_sparse)
                            p_plus.V = outer_square_vertices
                        elif p_plus_implementation == 'h_to_v_popcount_sparse':
                            B_initialization = np.array([
                                [1,1,0,0],
                                [1,0,1,0],
                                [0,0,1,1],
                                [0,1,0,1],
                            ], dtype=bool)
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
                    

                        np.random.seed(42)
                        for iter in range(n_iter):
                            print('iteration', iter)
                            p_plus, p_minus = iteration_loop_sphere(p_plus, p_minus)
                            outer_volume, inner_volume, average_volume = volume_estimate_convex_hull(p_plus, p_minus)
                            print(f'iteration {iter}: inner volume {inner_volume}, outer volume {outer_volume}, average volume {average_volume}')
                        
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
                            if not os.path.exists('tests/results/test_cube_in_sphere_in_cube'): # mkdir if not exists
                                os.makedirs('tests/results/test_cube_in_sphere_in_cube')
                            plt.savefig(f'tests/results/test_cube_in_sphere_in_cube/circle_approx_pplusimplementation_{p_plus_implementation.replace("_", "")}_pplususeDsparsedual{p_plus_use_D_sparse}_pminusimplementation_{p_minus_implementation.replace("_", "")}_pminusdualimplementation_{str(p_minus_dual_implementation).replace("_", "")}_pminususeDsparsedual{p_minus_use_D_sparse_dual}_iter_{iter}.png', dpi=200)
                            plt.close()

                        residuals = p_plus_outside_p_minus(p_plus, p_minus)
                        atol = 1e-6
                        assert np.all(residuals <= atol), f"Some points in p_plus are outside p_minus by more than {atol} for implementation {p_plus_implementation}, residuals: {np.max(residuals)}"

                        outer_volume, inner_volume, average_volume = volume_estimate_convex_hull(p_plus, p_minus)
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
    lower = -np.ones(d)
    upper = np.ones(d)
    
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

    A_plus, b_plus = np.array(p_plus.H[0]), np.array(p_plus.H[1])
    A_minus, b_minus = np.array(p_minus.H[0]), np.array(p_minus.H[1])
    
    # Monte Carlo for outer volume
    outer_volume = volume_monte_carlo(A_plus, b_plus, n_samples=100000)
    inner_volume = volume_monte_carlo(A_minus, b_minus, n_samples=100000)
    
    average_volume = (outer_volume + inner_volume) / 2
    return outer_volume, inner_volume, average_volume

def test_cube_in_sphere_in_cube(n_iter, p_plus_implementations, p_minus_implementations, p_minus_dual_implementations, n_dim=3):
    inner_square_vertices = make_inner_square_vertices(n_dim)
    outer_square_vertices = make_outer_square_vertices(n_dim)

    n_dim_volume = np.pi**(n_dim/2) / np.math.gamma(n_dim/2 + 1)

    for p_plus_implementation in p_plus_implementations:
        for p_minus_implementation in p_minus_implementations:
            dual_implementations = p_minus_dual_implementations if p_minus_implementation == 'v_to_h_dual' else [None]
            for p_minus_dual_implementation in dual_implementations:
                for p_plus_use_D_sparse in [True, False] if p_minus_dual_implementation in ['h_to_v_popcount', 'h_to_v_popcount_sparse'] else [None]:
                    for p_minus_use_D_sparse_dual in [True, False] if p_minus_dual_implementation in ['h_to_v_popcount', 'h_to_v_popcount_sparse'] else [None]:
                        print(f'Testing implementation {p_plus_implementation} with p_minus implementation {p_minus_implementation}...')
                        p_plus_base = DoubleDescription(implementation='cdd')
                        p_plus_base.add_V(outer_square_vertices) 
                        A, b = p_plus_base.H 
                        B_initialization = build_B_from_H_and_V(A, b, np.array(outer_square_vertices))
                        if p_plus_implementation == 'cdd':
                            p_plus = p_plus_base
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


                        np.random.seed(42)
                        for iter in range(n_iter):
                            print('iteration', iter)
                            p_plus_base, p_minus_base = iteration_loop_sphere(p_plus, p_minus)
                            p_plus, p_minus = iteration_loop_sphere(p_plus, p_minus)
                            print(f'p_plus vertices: {len(p_plus.V)}, p_plus halfspaces: {p_plus.H[0].shape[0]}, p_minus vertices: {len(p_minus.V)}, p_minus halfspaces: {p_minus.H[0].shape[0]}')
                            for vol_func, func_name in zip([volume_estimate_convex_hull, volume_estimate_fast], ['convex hull volume estimate', 'monte carlo volume estimate']):
                                outer_volume, inner_volume, average_volume = vol_func(p_plus, p_minus)
                                rel_error = abs(average_volume - n_dim_volume) / n_dim_volume
                                print(f'iteration {iter}: inner volume {inner_volume:<10.5f}, outer volume {outer_volume:<10.5f}, average volume {average_volume:<10.5f}, target volume {n_dim_volume:<10.5f}, rel error {rel_error:<10.5f}. Volume estimate method: {func_name}')

                            def canonicalized_V_A_b(p):
                                V_canonical = canonicalize_vertices(np.array(p.V))
                                A_canonical, b_canonical = canonicalize_halfspaces(np.array(p.H[0]), np.array(p.H[1]))
                                return V_canonical, A_canonical, b_canonical
                            
                            V_plus, A_plus, b_plus = canonicalized_V_A_b(p_plus)
                            V_plus_base, A_plus_base, b_plus_base = canonicalized_V_A_b(p_plus_base)
                            assert np.allclose(V_plus, V_plus_base), f'fail for plus implementation {p_plus_implementation}, p_plus vertices: {V_plus}, expected: {V_plus_base}'
                            assert np.allclose(A_plus, A_plus_base), f'fail for plus implementation {p_plus_implementation}, p_plus A: {A_plus}, expected: {A_plus_base}'
                            assert np.allclose(b_plus, b_plus_base), f'fail for plus implementation {p_plus_implementation}, p_plus b: {b_plus}, expected: {b_plus_base}' 
                            
                            V_minus, A_minus, b_minus = canonicalized_V_A_b(p_minus)
                            V_minus_base, A_minus_base, b_minus_base = canonicalized_V_A_b(p_minus_base)
                            assert np.allclose(V_minus, V_minus_base), f'fail for p_minus implementation {p_minus_implementation} and dual implementation {p_minus_dual_implementation}, p_minus vertices: {V_minus}, expected: {V_minus_base}'
                            assert np.allclose(A_minus, A_minus_base), f'fail for p_minus implementation {p_minus_implementation} and dual implementation {p_minus_dual_implementation}, p_minus A: {A_minus}, expected: {A_minus_base}'
                            assert np.allclose(b_minus, b_minus_base), f'fail for p_minus implementation {p_minus_implementation} and dual implementation {p_minus_dual_implementation}, p_minus b: {b_minus}, expected: {b_minus_base}'


                        residuals = p_plus_outside_p_minus(p_plus, p_minus)
                        atol = 1e-6
                        assert np.all(residuals <= atol), f"Some points in p_plus are outside p_minus by more than {atol} for implementation {p_plus_implementation}, residuals: {np.max(residuals)}"

                        outer_volume, inner_volume, average_volume = volume_estimate_fast(p_plus, p_minus)
                        tol = 0.01 * n_dim_volume
                        assert outer_volume >= inner_volume - tol, f"Outer volume {outer_volume} is not greater than inner volume {inner_volume} by at least {tol} for implementation {p_plus_implementation}, p_minus implementation {p_minus_implementation}, p_minus dual implementation {p_minus_dual_implementation}"
                        rel_error = abs(average_volume - n_dim_volume) / n_dim_volume
                        assert np.isclose(rel_error, 0.05, atol=0.1), f'average volume {average_volume} is not close enough to {n_dim_volume}, p_plus implementation {p_plus_implementation}, p_minus implementation {p_minus_implementation}, p_minus dual implementation {p_minus_dual_implementation}'

if __name__ == "__main__":
    # test_H_and_V_after_cut(['h_to_v_edges'], [ 'v_to_h_dual'], ['h_to_v_popcount_sparse','h_to_v_popcount', 'cdd',])
    # test_vol_ndim(n_iter=10, p_plus_implementations=['h_to_v_edges',], p_minus_implementations=['cdd'], p_minus_dual_implementations=[None], n_dim=4)
    test_cube_in_sphere_in_cube(25, p_plus_implementations(), p_minus_implementations(), p_minus_dual_implementations(), n_dim=3) 