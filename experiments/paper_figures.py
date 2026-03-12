import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import math
from xgw.hyperplane_approx import _run_approx_yield, projection, construct_basis_eij
from xgw.gromov_wasserstein_m_dist import _run_convex_yield
from xgw.frank_wolfe import _frank_wolfe_iter


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


def figure_random_pointcloud( d, n_points, seed=0):
    np.random.seed(seed)
    print(f'Number of points in simple marginals test: {n_points}')
    mu = nu = np.ones(n_points) / n_points

    space_x = np.random.randn(n_points,d)
    space_y = np.random.randn(n_points,d)

    return mu, nu, space_x, space_y, 


def bounding_box_convergence_rate_vol_2d(n_points, niter):
    
    P_plus_volumes = []
    P_minus_volumes = []
    objective_list = []
    iteration_list = list(range(1, niter+1))
    mu, nu, space_x, space_y = figure_random_pointcloud(2,n_points)
    
    for elem in tqdm(_run_approx_yield(mu, nu, space_x, space_y, emd_kwargs={}, niter = niter)):
        
        P_plus, P_minus, objective = elem

        P_plus_vol = volume_convex_hull_from_vertices(np.array(P_plus.V))
        try:
            P_minus_vol = volume_convex_hull_from_vertices(np.array(P_minus.V))
        except Exception as e:
            P_minus_vol = np.nan
        P_plus_volumes.append(P_plus_vol)
        P_minus_volumes.append(P_minus_vol)
        objective_list.append(objective)

    n_panels = 2
    plt.rc('font', size=12)
    fig, axes = plt.subplots(n_panels,1)
    axes[0].plot(iteration_list, P_plus_volumes, color='k', label=r'$P_\Pi^{+}$ volume')
    axes[0].plot(iteration_list, P_minus_volumes, color='r', label=r'$P_\Pi^{-}$ volume')
    axes[1].plot(range(1, len(objective_list) + 1), objective_list, color='blue', label=r'Upper bound of $H(P_\Pi^{+},P_\Pi^{-})$')
    for idx in range(n_panels):
        axes[idx].set_xlabel('Iteration')
        if idx == 0:
            axes[idx].set_ylabel('Volume')
        else:
            axes[idx].set_ylabel(f'Upper bound of \n Haussdorff distance')
            axes[idx].set_yscale('log')
        axes[idx].legend()
    # mkdir if not exists
    if not os.path.exists('experiments/figures'):
        os.makedirs('experiments/figures')
    fig.suptitle(f'Algorithm convergence, {n_points} points in marginals, dimension {2}' )
    plt.tight_layout()
    fig.savefig(f'experiments/figures/bounding_box_vol_d{2}_{n_points}points.svg', format='svg')
    plt.close(fig)  
    
def bounding_box_convergence_rate(n_points, niter, d):
    
    P_plus_size = []
    P_minus_size = []
    objective_list = []
    iteration_list = list(range(1, niter+1))
    mu, nu, space_x, space_y = figure_random_pointcloud(d, n_points)
    
    for elem in tqdm(_run_approx_yield(mu, nu, space_x, space_y, emd_kwargs={}, niter = niter)):
        
        P_plus, P_minus, objective = elem

        P_plus_size.append([len(P_plus.V), len(P_plus.H[0])])
        P_minus_size.append([len(P_minus.V), len(P_minus.H[0])])
        objective_list.append(objective)

    P_plus_size, P_minus_size = np.array(P_plus_size), np.array(P_minus_size)
    n_panels = 2
    plt.rc('font', size=12)
    fig, axes = plt.subplots(n_panels,1)
    axes[0].plot(iteration_list, P_plus_size[:,0], color='k', label=r'$P_\Pi^{+} vertices$')
    axes[0].plot(iteration_list, P_minus_size[:,1], color='r', label=r'$P_\Pi^{-}$ constraints')
    axes[1].plot(range(1, len(objective_list) + 1), objective_list, color='blue', label=r'Upper bound of $H(P_\Pi^{+},P_\Pi^{-})$')
    for idx in range(n_panels):
        axes[idx].set_xlabel('Iteration')
        if idx ==0:
            axes[idx].set_ylabel('Number')
        else:
            axes[idx].set_ylabel(f'Upper bound of \n Haussdorff distance')
            axes[idx].set_yscale('log')
        axes[idx].legend()
    # mkdir if not exists
    if not os.path.exists('experiments/figures'):
        os.makedirs('experiments/figures')
    fig.suptitle(f'Algorithm convergence, {n_points} points in marginals, dimension {d}' )
    plt.tight_layout()
    fig.savefig(f'experiments/figures/bounding_box_size_d{d}_{n_points}points.svg', format='svg')
    plt.close(fig)  
    



def convex_cost_convergence_rate(n_points, niter, d, cost):
    
    P_plus_size = []
    P_minus_size = []
    objective_list = []
    
    mu, nu, space_x, space_y = figure_random_pointcloud(d, n_points)
    
    for elem in tqdm(_run_convex_yield(mu, space_x, nu, space_y, emd_kwargs={}, niter = niter, cost=cost, p_plus_implementation= 'cdd')):
        
        P_plus, P_minus, objective = elem

        P_plus_size.append([len(P_plus.V), len(P_plus.H[0])])
        P_minus_size.append([len(P_minus.V), len(P_minus.H[0])])
        objective_list.append(objective)

    P_plus_size, P_minus_size = np.array(P_plus_size), np.array(P_minus_size)
    print(P_plus_size.shape,P_minus_size.shape )
    n_panels = 2
    plt.rc('font', size=12)
    fig, axes = plt.subplots(n_panels,1)
    iteration_list = list(range(1, min(niter, len(P_plus_size))+1))
    axes[0].plot(iteration_list, P_plus_size[:,0], color='k', label=r'$P_\Pi^{+} vertices$')
    axes[0].plot(iteration_list, P_minus_size[:,1], color='r', label=r'$P_\Pi^{-}$ constraints')
    axes[1].plot(range(1, len(objective_list) + 1), objective_list, color='blue', label=r' $ |c^+-c^*|$')
    for idx in range(n_panels):
        axes[idx].set_xlabel('Iteration')
        if idx ==0:
            axes[idx].set_ylabel('Number')
        else:
            axes[idx].set_ylabel(f'Bound on value \n '+r'of the GW$_m$ distance')
            axes[idx].set_yscale('log')
        axes[idx].legend()
    # mkdir if not exists
    if not os.path.exists('experiments/figures'):
        os.makedirs('experiments/figures')
    fig.suptitle(f'Algorithm convergence, {n_points} points in marginals, dimension {d}' )
    plt.tight_layout()
    fig.savefig(f'experiments/figures/{cost}_cost_convergence_d{d}_{n_points}points.svg', format='svg')
    plt.close(fig)  
    
    
    
    
def FW_convergence_rate(n_points, niter, d, cost):
    
    objective_list = []
    coupli_list = []
    mu, nu, space_x, space_y = figure_random_pointcloud(d, n_points)
    d = len(space_x[0])
    # e_base, R = construct_basis_eij(space_x, space_y)
    # p = 0.999
    # pi_init = p*np.outer(mu,mu)+(1-p)*np.diag(mu)
    # # print(d)
    # # np.random.seed(1)
    # # init_direc = np.random.rand(d**2)
    for elem in tqdm(_frank_wolfe_iter(mu, space_x, nu, space_y, cost=cost, iter_max=niter)):
        objective, coupling, _= elem
        objective_list.append(objective)
        coupli_list.append(coupling)
        final_plan = coupling
    # for elem in tqdm(_frank_wolfe_iter(mu, space_x, nu, space_y, cost=cost, iter_max=niter)):
    #     objective, coupling, _= elem
    #     coupli_list.append(np.linalg.norm(coupling-final_plan))
    # diff = -np.diff(np.array(objective_list))
    diff = np.array(objective_list)
    diff = diff-diff[-1]
    diff = diff[:-1]
    coupls = np.array(coupli_list)
    coupls = coupls - coupls[-1]
    coupls = coupls[:-1]
    coupls = [np.linalg.norm(val) for val in coupls ]
    
    n_panels = 2
    plt.rc('font', size=12)
    fig, axes = plt.subplots(n_panels,1)
    axes[0].plot(range(1, len(diff) + 1), coupls, color='k', label=r'Plan error')
    axes[1].plot(range(1, len(diff) + 1), diff, color='blue', label=r' Cost error')
    for idx in range(n_panels):
        axes[idx].set_xlabel('Iteration')
        if idx ==0:
            axes[idx].set_ylabel('Error')
        else:
            axes[idx].set_ylabel('Error')
            axes[idx].set_yscale('log')
        axes[idx].legend()
    # n_panels = 1
    # plt.rc('font', size=12)
    # fig, axes = plt.subplots(n_panels,1)
    # axes.plot(range(1, len(diff) + 1), diff, color='blue', label=r' $ \epsilon$')
    
    # axes.set_xlabel('Iteration')
    # axes.set_ylabel('Difference to local optimal cost')
    # axes.set_yscale('log')
    # axes.legend()
    # # mkdir if not exists
    if not os.path.exists('experiments/figures'):
        os.makedirs('experiments/figures')
    fig.suptitle(f'Frank-Wolfe convergence, \n {n_points} points in marginals, dimension {d}' )
    plt.tight_layout()
    fig.savefig(f'experiments/figures/FW_{cost}_cost_convergence_d{d}_{n_points}points.svg', format='svg')
    plt.close(fig)  
    
if __name__ == "__main__":
    # bounding_box_convergence_rate_vol_2d(10, 350)
    # bounding_box_convergence_rate_vol_2d(50, 500)
    # bounding_box_convergence_rate(10, 45, 3)
    # bounding_box_convergence_rate(10, 500, 2)
    # bounding_box_convergence_rate(50, 500, 2)
    # convex_cost_convergence_rate(50, 500, 2,'IGW')
    # convex_cost_convergence_rate(100, 500, 2,'IGW')
    # convex_cost_convergence_rate(5000, 500, 2,'IGW')
    # convex_cost_convergence_rate(10, 35, 3,'IGW')
    # FW_convergence_rate(1000, 29, 3,'DGW')
    FW_convergence_rate(1000, 200, 3,'IGW')
    # FW_convergence_rate(1000, 200, 3,'CGW')