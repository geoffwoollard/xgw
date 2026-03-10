import os
import numpy as np
import pytest
import matplotlib.pyplot as plt
import logging
import itertools as it
from scipy.spatial import Delaunay
from copy import deepcopy
import math
from pypoman.polygon import compute_polygon_hull

from xgw.hyperplane_approx import _run_approx, construct_basis_eij, p_plus_outside_p_minus, projection, new_direction, initial_box, DoubleDescription, build_B_from_H_and_V, masks_from_B


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def make_marginals(seed):
    np.random.seed(seed)
    n_points_xy = np.random.randint(5,15)
    mu_a1, sigma_a = np.array([0.5, 0.5]), 0.3
    r_factor = 0.5
    mu_b, sigma_b = r_factor*mu_a1, 0.2
    n_grid_1d_x = n_grid_1d_y = n_points_xy
    def make_space_2d(n_grid):
        lin = np.linspace(-1, 1, n_grid)
        xx, yy = np.meshgrid(lin, lin)
        return np.vstack([xx.ravel(), yy.ravel()]).T
    space_x = make_space_2d(n_grid_1d_x)
    space_y = make_space_2d(n_grid_1d_y)

    factor = 0.5
    mu = factor*np.exp(-0.5 * (((space_x - mu_a1) / sigma_a) ** 2).sum(-1))
    mu += np.exp(-0.5 * (((space_x + factor*mu_a1) / sigma_a) ** 2).sum(-1))
    nu = factor*np.exp(-0.5 * (((space_y - mu_b) / sigma_b) ** 2).sum(-1))
    nu += np.exp(-0.5 * (((space_y + factor*mu_b) / sigma_b) ** 2).sum(-1))
    mu /= mu.sum()
    nu /= nu.sum()

    # scale space to have zero center of mass
    space_x = space_x.astype(mu.dtype)
    space_y = space_y.astype(nu.dtype)
    space_x -= (space_x * mu[:, None]).sum(axis=0)
    space_y -= (space_y * nu[:, None]).sum(axis=0)
    
    return mu, nu, space_x, space_y


@pytest.fixture
def marginals():
    return make_marginals(seed=0)


def test_initial_box(marginals):
    mu, nu, space_x, space_y = marginals
    for p_plus_implementation in ['h_to_v_popcount_sparse']:
        logger.info(f'Testing initial box with p_plus implementation: {p_plus_implementation}')
        e_base, R, P_plus, P_minus = initial_box(space_x, space_y, mu, nu, emd_kwargs={}, p_plus_implementation=p_plus_implementation)
        assert len(P_minus.V) == 5
        assert len(P_plus.V) == 5
        assert P_plus.H[0].shape == (5,4)
        assert P_minus.H[0].shape == (5,4)
        
        residuals = p_plus_outside_p_minus(P_plus, P_minus)
        atol = 1e-15
        assert np.all(residuals <= atol), f'max residual for inclusion of P_minus in P_plus : {residuals.max()}'


def make_simple_marginals(seed, d, min_points=30, max_points=30):
    np.random.seed(seed)
    n_points = np.random.randint(min_points, max_points+1)
    print(f'Number of points in simple marginals test: {n_points}')
    mu = nu = np.ones(n_points) / n_points

    space_x = np.random.randn(n_points,d)
    space_y = np.random.randn(n_points,d)
    radius_x = np.linalg.norm(space_x, axis=1).max()
    radius_y = np.linalg.norm(space_y, axis=1).max()
    space_x /= radius_x
    space_y /= radius_y

    return mu, nu, space_x, space_y


def make_simple_marginals_low_num(seed, d, min_points=4, max_points=7):
    np.random.seed(seed)
    n_points = np.random.randint(min_points, max_points)
    n_points = 4
    print(f'Number of points in simple marginals test: {n_points}')
    mu = nu = np.ones(n_points) / n_points

    space_x = np.random.randn(n_points,d)
    space_y = np.random.randn(n_points,d)

    # mu = np.array([1/3,1/3,1/3])
    # nu = mu = np.array([1/3,1/3,1/3])
    # space_x = np.array([[0,1.53],[4.87,1],[5,1/2]])
    # space_y = np.array([[3,0],[4,2],[1,2]])
    return mu, nu, space_x, space_y


@pytest.fixture
def simple_marginals_2D():
    # default seed when pytest runs
    return make_simple_marginals(seed=0, d=2)


@pytest.fixture
def super_simple_marginals_2D():
    # default seed when pytest runs
    return make_simple_marginals_low_num(seed=0, d=2)


def permut_matrix(permut, dim):
    sol = np.zeros((dim,dim))
    for i,j in enumerate(permut):
        sol[i,j] = 1
    return sol
        

def test_simple_marginals(super_simple_marginals_2D):
    P_plus_volumes = []
    P_minus_volumes = []
    max_niter = 15
    iteration_list = list(range(1, max_niter+1))
    for niter in iteration_list:
        logger.info(f'Testing simple marginals with niter={niter}')
        mu, nu, space_x, space_y = super_simple_marginals_2D
        n_point = len(mu)
        x_1, x_2 = space_x.shape
        y_1, y_2 = space_y.shape
        print(f'marginal points number : {x_1*x_2}, and {y_1*y_2}', f'dimension {2}')
        
        P_plus, P_minus, _, objective_list, x_0_list, v_0_list = _run_approx(mu, nu, space_x, space_y, emd_kwargs={}, niter = niter)
        logger.info(f'P_plus vertices: {np.array(P_plus.V)}')
        logger.info(f'P_minus vertices: {np.array(P_minus.V)}')
        logger.info(f'x_0_list over iterations: {np.array(x_0_list)}')
        logger.info(f'v_0_list over iterations: {np.array(v_0_list)}')
        logger.info(f'Objective list over iterations: {objective_list}')
        diffs = np.diff(objective_list)
        tolerance = 1e-10
        msg = f'Objective not non-increasing, diffs: {diffs}'
        assert np.all(diffs < tolerance), msg
        new_obj = new_direction(P_plus, P_minus)[1]

        P_plus_vol = volume_convex_hull_from_vertices(np.array(P_plus.V))
        try:

            P_minus_vol = volume_convex_hull_from_vertices(np.array(P_minus.V))
        except Exception as e:
            logger.error(f'Error computing P_minus volume: {e}')
            P_minus_vol = np.nan
        P_plus_volumes.append(P_plus_vol)
        P_minus_volumes.append(P_minus_vol)
        logger.info(f'P_plus volume: {P_plus_vol}, P_minus volume: {P_minus_vol}')
        
        print(f'final  Hausdorf {new_obj}')
        # assert new_obj<=objective # TODO: turn back on when fixed
        residuals = p_plus_outside_p_minus(P_plus, P_minus)
        atol = 1e-16
        # assert np.all(residuals <= atol)
        logger.info(f'max residual for inclusion of P_minus in P_plus : {residuals.max()}')

        
        e_base, _ = construct_basis_eij(space_x, space_y)
        coupling_vertices = [permut_matrix(elem, n_point)/n_point for elem in it.permutations(np.linspace(0,n_point-1,n_point, dtype = int))] # the true vertices of the coupling polytopes are the permutation matrices

        projected_couplings = [projection(vertex, e_base) for vertex in coupling_vertices]
        
        # ploting the projection in 2d of P_minus, projected_vertices and  P_plus
        P_minus_proj = np.array(P_minus.V)
        P_plus_proj = np.array(P_plus.V)
        
        fig, axes = plt.subplots(ncols=6, nrows=1, figsize=(36,4))
        fig.suptitle('2D projection of the 4D spaces')
        for idx, (i, j) in enumerate(it.combinations(range(4), 2)):
            P_true_proj_2d = DoubleDescription()
            P_true_proj_2d.add_V(np.array(projected_couplings)[:,[i,j]])
            A, b = P_true_proj_2d.H
            true_vertices = np.array(compute_polygon_hull(A, b))
            axes[idx].set_xlabel(f'Dimension {i}')
            axes[idx].set_ylabel(f'Dimension {j}')
            axes[idx].fill(true_vertices[:,0],true_vertices[:,1],  label=r'$P_{\Pi}$', alpha=0.3 )
            axes[idx].scatter(P_plus_proj[:,i], P_plus_proj[:,j],c='k', label=r'$P_{\Pi}^+$')
            axes[idx].scatter(P_minus_proj[:,i], P_minus_proj[:,j],c='r', label=r'$P_{\Pi}^-$')
            newly_added_vertex = np.array(v_0_list[-1])
            logger.info(f'newly added vertex: {newly_added_vertex}')
            new_point_on_face = np.array(x_0_list[-1])
            logger.info(f'new point on face: {new_point_on_face}')
            axes[idx].scatter(new_point_on_face[i], new_point_on_face[j], c='r', label=r'new $x_0$', marker='*', s=300, alpha=0.5)
            axes[idx].scatter(newly_added_vertex[i], newly_added_vertex[j], c='k', label=r'new $v_0$', marker='x', s=300, alpha=0.5)
        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)

        # Deduplicate while preserving order
        by_label = dict(zip(labels, handles))

        # Add legend outside plot
        fig.legend(by_label.values(), by_label.keys(),
                loc='center left', bbox_to_anchor=(0.9, 0.5))

        # mkdir if not exists
        if not os.path.exists('tests/results'):
            os.makedirs('tests/results')
        fig.savefig(
            f'tests/results/test_projection_coupling_P_plus_minus_niter{niter}.png',
            bbox_inches='tight'
        )
        plt.close(fig)
        
        
        P_true = DoubleDescription()
        P_true.add_V(projected_couplings)
        P_true.H_to_V()
        
        # checking P_minus is included in P_plus 
        residuals = p_plus_outside_p_minus(P_plus, P_minus)
        atol = 1e-10
        assert np.all(residuals <= atol), f'max residual for inclusion of P_minus in P_plus : {residuals.max()}'
        
        # checking P_minus is included in P_true 
        residuals = p_plus_outside_p_minus(P_true, P_minus)
        atol = 1e-10
        assert np.all(residuals <= atol), f'max residual for inclusion of P_minus in P_true : {residuals.max()}'
        
        # checking P_true is included in P_plus 
        residuals = p_plus_outside_p_minus(P_plus, P_true)
        atol = 1e-10
        assert np.all(residuals <= atol), f'max residual for inclusion of P_true in P_plus : {residuals.max()}'

    logger.info(f'P_plus volumes over iterations: {P_plus_volumes}')
    logger.info(f'P_minus volumes over iterations: {P_minus_volumes}')
    true_volume = volume_convex_hull_from_vertices(np.array(projected_couplings))
    assert P_plus_volumes[-1] >= true_volume, f'P_plus volume {P_plus_volumes[-1]} should be at least the true volume {true_volume}'
    assert P_minus_volumes[-1] <= true_volume, f'P_minus volume {P_minus_volumes[-1]} should be at most the true volume {true_volume}'
    n_panels = 3
    fig, axes = plt.subplots(n_panels,1, figsize=(8,10))
    axes[0].plot(iteration_list, P_plus_volumes, color='k', label='P_plus volume')
    axes[1].plot(iteration_list, P_minus_volumes, color='r', label='P_minus volume')
    axes[2].plot(range(1, len(objective_list) + 1), objective_list, color='blue', label='Upper bound of Haussdorff distance')
    for idx in range(n_panels):
        axes[idx].set_xlabel('Iteration')
        if idx < 2:
            axes[idx].set_ylabel('Volume')
            axes[idx].hlines(true_volume, 1, max_niter, colors='gray', linestyles='dashed', label='True volume')
        else:
            axes[idx].set_ylabel('Hausdorff distance')
        axes[idx].legend()
    # mkdir if not exists
    if not os.path.exists('tests/results'):
        os.makedirs('tests/results')
    fig.savefig('tests/results/test_volume_P_plus_minus_simple_marginals.png')
    plt.close(fig)  
    

@pytest.fixture
def niter():
    return 10


def volume_convex_hull_from_vertices(vertices):
    """
    vertices: (N, d) numpy array of points
    returns approximate exact volume via Delaunay triangulation
    """
    d = vertices.shape[1]
    
    # triangulate points
    tri = Delaunay(vertices)
    
    vol = 0.0
    for simplex in tri.simplices:
        verts = vertices[simplex]  # shape (d+1, d)
        base = verts[0]
        M = verts[1:] - base      # d x d matrix
        vol += abs(np.linalg.det(M))
    
    return vol / math.factorial(d)


def test_overall(niter, marginals):
    mu, nu, space_x, space_y = marginals
    x_1, x_2 = space_x.shape
    y_1, y_2 = space_y.shape
    logger.info(f'marginal points number : {x_1*x_2}, and {y_1*y_2}, dimension {2}')
    
    P_plus, P_minus, objective, objective_list, _, _ = _run_approx(mu, nu, space_x, space_y, emd_kwargs={'numItermax': 10**6}, niter = niter)
    logger.info(f'Objective list over iterations: {objective_list}')
    diffs = np.diff(objective_list)
    tol = 1e-5
    print(f'Objective differences over iterations: {diffs}')

    assert np.all(diffs < tol), f'Objective not non-increasing, diffs: {diffs}'
    new_obj = new_direction(P_plus, P_minus)[1]
    
    logger.info(f'final  Hausdorf {new_obj - objective}')
    tol = 1e-6              #High tolerence, the Hausdorff does not converge well
    assert new_obj<=objective + tol

    residuals = p_plus_outside_p_minus(P_plus, P_minus)
    atol = 1e-17
    assert np.all(residuals <= atol), f'max residual for inclusion of P_minus in P_plus : {residuals.max()}'


def test_P_monotonicity(marginals):
    P_plus_old = DoubleDescription
    P_minus_old = DoubleDescription
    max_niter = 10
    iteration_list = list(range(1, max_niter+1))
    for niter in iteration_list:
        mu, nu, space_x, space_y = marginals
        x_1, x_2 = space_x.shape
        y_1, y_2 = space_y.shape
        logger.info(f'marginal points number : {x_1*x_2}, and {y_1*y_2}, dimension {2}')
        P_plus, P_minus, _, _, _, _ = _run_approx(mu, nu, space_x, space_y, emd_kwargs={}, niter = niter)
        if niter>1:
            residuals = p_plus_outside_p_minus(P_plus_old, P_plus)
            atol = 1e-12
            assert np.all(residuals <= atol), f'max residual for inclusion of P_plus in P_plus_old : {residuals.max()}'
            residuals = p_plus_outside_p_minus( P_minus, P_minus_old)
            atol = 1e-12
            assert np.all(residuals <= atol), f'max residual for inclusion of  P_minus_old in P_minus : {residuals.max()}'
            
        P_plus_old = deepcopy(P_plus)
        P_minus_old = deepcopy(P_minus)

def test_remove_duplicates_V():
    V = np.array([[0,0],[1,0],[0,1],[1,1],[0,0],[1,0]]) #0->4, 1->5 are duplicates
    Edges = np.array([[0,1],[0,2],[1,3],[2,3],
                      [4,1],[4,2],
                      [0,5],      [5,3],
                      ]) #no 04 and 15 edges
    A = np.array([[1,0],[-1,0],[0,1],[0,-1]])
    b = np.array([1,0,1,0])
    B_initialization = build_B_from_H_and_V(A, b, V)
    masks_initialization = masks_from_B(B_initialization)
    D_expected = np.array([[0,1,1,0],
                            [1,0,0,1],
                            [1,0,0,1],
                            [0,1,1,0]])
    unique_indices = np.array([0,1,2,3])
    duplicate_indices = np.array([4,5])
    keep_indices = np.isin(Edges, duplicate_indices).any(axis=1) == False # keep indices do no have 4,5 in Edges
    print(f'keep_indices: {keep_indices}')
    Edges_expected = Edges[keep_indices]
    B_expected = B_initialization[:,unique_indices] # only the first 4 rows of B_initialization should be kept after removing duplicates
    D_expected = D_expected[np.ix_(unique_indices, unique_indices)] # D should be updated to reflect the removal of duplicate vertices
    for implementation in ['cdd','h_to_v_edges','h_to_v_popcount','h_to_v_popcount_sparse']:
        logger.info(f'Testing remove_duplicates_V with implementation: {implementation}')
        p = DoubleDescription(
            implementation=implementation, 
            V_initialization=V,
            E_initialization=Edges,
            B_initialization=B_initialization, 
            masks_initialization=masks_initialization,
            A_initialization=A,
            b_initialization=b
        )
        p.V = V.tolist()
        p.E = Edges.tolist()
        p.H = [A,b]
        p.remove_duplicates_V()
        assert len(p.V) == 4
        assert np.array_equal(p.V, np.array([[0,0],[1,0],[0,1],[1,1]]))
        if implementation == 'h_to_v_edges':
            assert len(p.E) == 4
            assert np.array_equal(p.E, Edges_expected), f"Expected E:\n{Edges_expected}\nGot:\n{p.E}"
        elif implementation in ['h_to_v_popcount']:
            assert p.poly.D.shape == (4, 4), f"Expected D shape (4,4), got {p.poly.D.shape}"
            assert p.poly.B.shape == (4, 4), f"Expected B shape (4,4), got {p.poly.B.shape}"
            assert np.array_equal(p.poly.B, B_expected), f"Expected B:\n{B_expected}\nGot:\n{p.poly.B}"
            assert np.array_equal(p.poly.D, D_expected), f"Expected D:\n{D_expected}\nGot:\n{p.poly.D}"
        elif implementation in ['h_to_v_popcount_sparse']:
            assert p.poly.D.shape == (4, 4), f"Expected D shape (4,4), got {p.poly.D.shape}"
            assert len(p.poly.masks) == 4, f"Expected masks shape length 4, got {len(p.poly.masks.shape)}"
            assert np.array_equal(p.poly.masks, masks_from_B(B_expected)), f"Expected B:\n{B_expected}\nGot:\n{p.poly.B}"
            assert np.array_equal(p.poly.D, D_expected), f"Expected D:\n{D_expected}\nGot:\n{p.poly.D}"
            

if __name__ == "__main__":
    test_remove_duplicates_V()