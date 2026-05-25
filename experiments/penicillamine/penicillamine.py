from copy import deepcopy
import itertools
import os
import hydra
from hydra.core.hydra_config import HydraConfig
from rdkit import Chem
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from dataclasses import dataclass
from multiprocessing import Pool
import itertools
from scipy.stats import ks_2samp, mannwhitneyu

from xgw.gromov_wasserstein_m_dist import gw_m_convex
from xgw.evalulate_fixed_plan import evaluate_cost, evaluate_cost_precompute, evaluate_cost_postcompute

def read_molecule(fname):

    supplier = Chem.SDMolSupplier(fname)

    mol = supplier[0]
    conf = mol.GetConformer()

    coords = []
    atoms = []

    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])
        atoms.append(atom.GetSymbol())
    return coords, atoms

def center_and_normalize(coords):
    points = np.array(coords)
    points -= points.mean(axis=0)
    points /= np.linalg.norm(points, axis=1, keepdims=True).max()
    points
    return points

def make_conformers(points, n_conformers, noise_level):
    conformers = np.tile(points, (n_conformers, 1))
    conformers = conformers.reshape(n_conformers, -1, 3)
    noise = noise_level * np.random.randn(*conformers.shape)
    conformers += noise
    return conformers, noise

def optimize_by_enumeration(conformer_1, conformer_2, n_points, t):
    best_plan = None
    best_cost = float('inf')
    id_plan = np.eye(n_points) / n_points
    mu = nu = np.ones(n_points) / n_points
    cst_cost, e_base, R, d, t = evaluate_cost_precompute(conformer_1, conformer_2, mu, nu, 'CGW', t)
    for perm in itertools.permutations(range(n_points)):
        plan = id_plan[list(perm)] 
        total_cost = evaluate_cost_postcompute(e_base, R, d, t, plan, 'CGW', cst_cost)
        if total_cost < best_cost:
            best_cost = total_cost
            best_plan = id_plan[list(perm)]
    return best_cost, best_plan

def evaluate_permutation_static(args):
    """Static function for multiprocessing (must be at module level)."""
    perm, id_plan, e_base, R, d, t, cst_cost = args
    plan = id_plan[list(perm)]
    total_cost = evaluate_cost_postcompute(e_base, R, d, t, plan, 'CGW', cst_cost)
    return total_cost, perm

def optimize_by_enumeration_pool(conformer_1, conformer_2, n_points, t, n_processes):
    """Parallelize with multiprocessing.Pool."""
    best_plan = None
    best_cost = float('inf')
    id_plan = np.eye(n_points) / n_points
    mu = nu = np.ones(n_points) / n_points
    cst_cost, e_base, R, d, t = evaluate_cost_precompute(conformer_1, conformer_2, mu, nu, 'CGW', t)
    
    # Generate all permutations
    perms = list(itertools.permutations(range(n_points)))
    
    # Create argument tuples
    args = [(perm, id_plan, e_base, R, d, t, cst_cost) for perm in perms]
    
    # Parallel evaluation
    with Pool(n_processes) as pool:
        results = pool.map(evaluate_permutation_static, args)
    
    # Find best result
    for total_cost, perm in results:
        if total_cost < best_cost:
            best_cost = total_cost
            best_plan = id_plan[list(perm)]

    print(f'Best cost found: {best_cost}')
    print(f'Best permutation: {perm}')
    
    return best_cost, best_plan

def plan_to_permutation(plan):
    """Extract permutation from plan matrix (identity permutation with reordering)."""
    # Find the column index with max value (0.11111111) for each row
    perm = np.argmax(plan, axis=1)
    return tuple(perm)

def run(fname_molecule_input, fname_output, iter_max, n_conformers, noise_level_config, flip, solution_method, n_processes, random_seed):
    np.random.seed(random_seed)
    coords, _ = read_molecule(fname_molecule_input)
    points = center_and_normalize(coords)
    noise_levels = np.linspace(noise_level_config.min, noise_level_config.max, n_conformers)
    conformers, noise = np.empty((n_conformers, len(points), 3)), np.empty((n_conformers, len(points), 3))
    for idx, noise_level in enumerate(noise_levels):
        _conformers, _noise = make_conformers(points, n_conformers=1, noise_level=noise_level)
        conformers[idx] = _conformers
        noise[idx] = _noise
        
    conformers = center_and_normalize(conformers.reshape(-1, 3)).reshape(conformers.shape)
    r2_max_after = np.linalg.norm(conformers.reshape(-1, 3), axis=1).max()**2
    t = 8*r2_max_after / (2 + 8*r2_max_after) 

    mu = np.ones(len(points)) / len(points)

    cgw2s = np.zeros((n_conformers, n_conformers))
    rmsds = np.zeros((n_conformers, n_conformers))
    rmsds_otalignment = np.zeros((n_conformers, n_conformers))
    plans = np.zeros((n_conformers, n_conformers, len(points), len(points)))
    permutations = np.zeros((n_conformers, n_conformers, len(points)), dtype=object)
    for i in range(n_conformers):
        for j in range(i, n_conformers):    
            if j >= i:        
                conformer_j = deepcopy(conformers[j])
                if flip:
                    conformer_j[:, 0] *= -1

                if solution_method == 'polytope':
                    cgw2, plan, _, _, _, _, _ = gw_m_convex(mu, conformers[i], mu, conformer_j, {'numItermax': 10**9}, cost='CGW', gap_tol=1e-15, iter_max=iter_max, t=t, FW_iter=100,
                                                        p_plus_implementation='h_to_v_popcount_sparse', 
                                                        p_minus_implementation='v_to_h_dual', 
                                                        p_minus_dual_implementation='h_to_v_popcount_sparse') 
                elif solution_method == 'enumeration':                        
                    cgw2, plan = optimize_by_enumeration(conformers[i], conformer_j, len(points), t)
                elif solution_method == 'enumeration_pool':                        
                    cgw2, plan = optimize_by_enumeration_pool(conformers[i], conformer_j, len(points), t, n_processes)

                elif solution_method == 'id':
                    plan = np.eye(len(points)) / len(points)
                    cgw2 = evaluate_cost(conformers[i], conformer_j, plan, 'CGW', t)
                else:
                    raise ValueError(f'Unknown solution method: {solution_method}')

                cgw2s[i, j] = cgw2s[j, i] = cgw2
                plans[i, j] = plans[j, i] = plan

                # Compute RMSD
                optimal_alignment = conformer_j[plan.argmax(axis=1)]
                rmsds_otalignment[i, j] = rmsds_otalignment[j, i] = np.linalg.norm(conformers[i] - optimal_alignment) / np.sqrt(len(points))

                rmsd = np.linalg.norm(conformers[i] - conformer_j) / np.sqrt(len(points))
                rmsds[i, j] = rmsds[j, i] = rmsd
                permutations[i,j] = permutations[j, i] = plan_to_permutation(plan)

    np.savez(fname_output, conformers=conformers, noise=noise, cgw2s=cgw2s, rmsds=rmsds, plans=plans, rmsds_otalignment=rmsds_otalignment)

def plot(fname, odir):
    data = np.load(fname)
    cgw2s = data['cgw2s']
    cgws = np.sign(cgw2s) * np.sqrt(np.abs(cgw2s))
    rmsds = data['rmsds']

    plt.rcParams.update({
        'figure.figsize': (12, 10),
        'font.size': 16,
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 16,
        'lines.linewidth': 2.5,
        'axes.linewidth': 2,
    })

    plt.imshow(rmsds, cmap='gray')
    plt.title('RMSD Distances')
    plt.colorbar()
    plt.savefig(os.path.join(odir, 'rmsd.png'))
    plt.clf()

    plt.imshow(cgws, cmap='gray')
    plt.title('CGW Distances')
    plt.colorbar()
    plt.savefig(os.path.join(odir, 'cgw.png'))
    plt.clf()

    plt.scatter(cgws.flatten(), rmsds.flatten())
    plt.xlabel('CGW Distance')
    plt.ylabel('RMSD Distance')
    plt.title('CGW vs RMSD')
    plt.savefig(os.path.join(odir, 'cgw_vs_rmsd.png'))
    plt.clf()

    # mismatch from id
    plans = data['plans']
    traces = np.einsum('ijkk->ij', plans)
    plt.imshow(traces, cmap='gray')
    plt.title('Trace of Optimal Transport Plans')
    plt.colorbar()
    plt.savefig(os.path.join(odir, 'trace.png'))
    plt.clf()

    df = pd.DataFrame({'CGW': cgws.flatten(), 'RMSD': rmsds.flatten()})
    df.to_csv(os.path.join(odir, 'cgw_rmsd.csv'), index=False)
    
def plot_flip_vs_non_flip(fname_flip, fname_non_flip, odir):
    # Or configure matplotlib directly without seaborn
    plt.rcParams.update({
        'figure.figsize': (12, 10),
        'font.size': 16,
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 16,
        'lines.linewidth': 2.5,
        'axes.linewidth': 2,
    })
    data_flip = np.load(fname_flip)
    data_non_flip = np.load(fname_non_flip)

    rmsd_scale = 1 #1000
    cgw_scale = 1 # 10**6
    cgws_flip = data_flip['cgws'] * cgw_scale
    rmsds_flip = data_flip['rmsds'] * rmsd_scale
    cgws_non_flip = data_non_flip['cgws'] * cgw_scale
    rmsds_non_flip = data_non_flip['rmsds'] * rmsd_scale

    _, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(y=cgws_non_flip.flatten(), x=rmsds_non_flip.flatten(), alpha=1, s=50, color='black')
    axis.set_ylabel('CGW')
    axis.set_xlabel('RMSD')
    plt.tight_layout()
    plt.savefig(os.path.join(odir, 'cgw_vs_rmsd.pdf'), dpi=300)
    plt.clf()

    _, axis = plt.subplots(figsize=(5, 5))

    # Boxplots
    upper_triangle_indices = np.triu_indices_from(cgws_flip, k=1)
    
    bp = axis.boxplot(
        [cgws_non_flip[upper_triangle_indices].flatten(), 
         cgws_flip[upper_triangle_indices].flatten()], 
        labels=['Without Reflection', 'With Reflection'],
        widths=0.6,
    )
    axis.set_ylabel('CGW Distance')
    # axis.set_title('Boxplot Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(odir, 'cgw_hist.pdf'), dpi=300)
    plt.clf()

    ks_statistic, ks_p_value = ks_2samp(cgws_non_flip[upper_triangle_indices].flatten(), 
         cgws_flip[upper_triangle_indices].flatten())

    mw_stat, mw_pval = mannwhitneyu(cgws_non_flip[upper_triangle_indices].flatten(), 
            cgws_flip[upper_triangle_indices].flatten())
    
    return {'ks_statistic': ks_statistic, 'ks_p_value': ks_p_value, 'mw_stat': mw_stat, 'mw_pval': mw_pval}


@dataclass
class NoiseLevelConfig:
    min: float
    max: float

@dataclass
class Config:
    fname_molecule_input: str 
    n_conformers: int 
    noise_level: NoiseLevelConfig 
    iter_max: int | None
    output_fname: str 
    flip: bool 
    solution_method: str
    n_processes: int
    random_seed: int

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: Config):
    print(config)
    n_conformers = config.n_conformers
    noise_level = config.noise_level
    iter_max = config.iter_max

    # hydra working directory
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    fname_molecule_input = config.fname_molecule_input
    output_fname = os.path.join(output_dir, config.output_fname)
    flip = config.flip
    run(fname_molecule_input, output_fname, iter_max, n_conformers, noise_level, flip, config.solution_method, config.n_processes, config.random_seed)
    odir = output_fname.replace('.npz', '_output')
    if not os.path.exists(odir):
        os.makedirs(odir)
    plot(output_fname, odir)



if __name__ == "__main__":
    main()