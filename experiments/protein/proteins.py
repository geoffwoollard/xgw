import gemmi
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import hydra
import ot
import logging
import os
from dataclasses import dataclass, asdict

# logging.disable(logging.CRITICAL)
# import logging
# logger = logging.getLogger(__name__)

# import sys

# # Clear existing handlers (important with Hydra)
# root = logging.getLogger()
# root.handlers.clear()

# # Send logs to stderr
# handler = logging.StreamHandler(sys.stderr)
# formatter = logging.Formatter(
#     "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
# )
# handler.setFormatter(formatter)

# root.addHandler(handler)
# root.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,  # or DEBUG
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from xgw.gromov_wasserstein_m_dist import gw_m_convex
from xgw.utils import gw_matrix, safe_plot


def extract_coords_and_atoms(fname, chosen_altloc='A'):
    
    st = gemmi.read_structure(fname)
    model_coords = []
    model_atoms = []
    for idx_model in range(len(st)):
        coords = []
        atoms = []
        for chain in st[idx_model]:
            for res in chain:
                for atom in res:
                    if atom.has_altloc() and atom.altloc != chosen_altloc:
                        continue
                    coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                    atoms.append((chain.name, res.seqid.num, atom.name))

        model_coords.append(np.array(coords))   # shape (N, 3)
        model_atoms.append(np.array(atoms))

    model_coords = np.array(model_coords)  # shape (M, N, 3)
    model_atoms = np.array(model_atoms)

    return model_coords, model_atoms

def selection_atoms(selection, model_coords, model_atoms):
    atom_type_idx = 2
    # idx = (model_atoms[:,:,atom_type_idx] in selection)
    mask = np.isin(model_atoms[:,:,atom_type_idx], selection)
    coords = model_coords[mask]
    return coords.reshape(model_coords.shape[0], -1, 3)


@dataclass
class Config:
    repo_dir: str
    fname_1: str 
    fname_2: str
    gw_plot_fname: str 
    selection: list[str] 
    n_models: int | None 
    n_skip_every: int
    compute_gw: bool
    compute_xgw: bool
    xgw_plot_fname: str 
    max_iter: int 
    unique_label: str
    dimension: int 
    odir: str 
    t_eps: float
    p_plus_implementation: str
    p_minus_implementation: str
    p_minus_dual_implementation: str
    flip: bool

@hydra.main(version_base=None, config_path=".", config_name="proteins_config")
def main(config: Config):
    model_coords_1, model_atoms_1 = extract_coords_and_atoms(config.fname_1)
    label_1 = os.path.basename(config.fname_1).replace('.cif', '')
    np.savez(config.fname_1.replace('.cif', '_models.npz'), model_coords=model_coords_1, model_atoms=model_atoms_1)

    model_coords_2, model_atoms_2 = extract_coords_and_atoms(config.fname_2)
    label_2 = os.path.basename(config.fname_2).replace('.cif', '')
    np.savez(config.fname_2.replace('.cif', '_models.npz'), model_coords=model_coords_2, model_atoms=model_atoms_2)

    n_models = config.n_models
    dimension = config.dimension
    coords_ca_1 = selection_atoms(config.selection, model_coords_1[:n_models], model_atoms_1[:n_models])[:,:,:dimension]
    coords_ca_2 = selection_atoms(config.selection, model_coords_2[:n_models], model_atoms_2[:n_models])[:,:,:dimension]

    coords_ca_1 = coords_ca_1 - np.mean(coords_ca_1, axis=1, keepdims=True)
    coords_ca_2 = coords_ca_2 - np.mean(coords_ca_2, axis=1, keepdims=True)
    r2_max = max(np.max(np.sum(coords_ca_1**2, axis=-1)), np.max(np.sum(coords_ca_2**2, axis=-1)))
    coords_ca_1 = coords_ca_1 / np.sqrt(r2_max)
    coords_ca_2 = coords_ca_2 / np.sqrt(r2_max)

    n_skip_every = config.n_skip_every
    flip = config.flip

    if config.compute_gw:
        cross_gw_losses = gw_matrix(coords_ca_1, coords_ca_2, symmetric=False, n_skip=n_skip_every, flip=flip)
        self_gw_losses_1 = gw_matrix(coords_ca_1, coords_ca_1, symmetric=True, n_skip=n_skip_every, flip=flip)
        self_gw_losses_2 = gw_matrix(coords_ca_2, coords_ca_2, symmetric=True, n_skip=n_skip_every, flip=flip)

        fig = plt.figure(figsize=(12, 12))  
        plt.boxplot([cross_gw_losses.flatten(), self_gw_losses_1.flatten(), self_gw_losses_2.flatten()])
        plt.xticks([1, 2, 3], ['cross', f'self_{label_1}', f'self_{label_2}'])
        # plt.yscale('log')
        plt.ylabel('GW loss')
        plt.savefig(config.gw_plot_fname.replace('.png', f'_{config.unique_label}.png'))
        plt.close(fig)
        np.savez(config.gw_plot_fname.replace('.png', f'_data_{config.unique_label}.npz'), 
                 config=asdict(config),
                 cross_gw_losses=cross_gw_losses,
                 self_gw_losses_1=self_gw_losses_1,
                 self_gw_losses_2=self_gw_losses_2)

    if config.compute_xgw:
        r2_x = np.linalg.norm(coords_ca_1, axis=(-1)).max()
        r2_y = np.linalg.norm(coords_ca_2, axis=(-1)).max()
        r2_max = max(r2_x, r2_y)
        if dimension == 3:
            t = 8*r2_max / (2 + 8*r2_max)
        elif dimension == 2:
            t = 0.5
        else:
            raise ValueError(f'dimension {dimension} must be in 2 or 3')
        t += config.t_eps
        
        print(f"Using t={t:.4f} based on r2_max={r2_max:.4f}")
        dpi = 300

        def gw_m_convex_wrapper(coords_ca_1, coords_ca_2, t, symmetric, iter_max, flip):
            losses_upper_bound = np.zeros((len(coords_ca_1), len(coords_ca_2)), dtype=float)
            losses_lower_bound = np.zeros_like(losses_upper_bound)
            losses_gap = np.zeros_like(losses_upper_bound)
            losses_constant = np.zeros_like(losses_upper_bound)
            plans = np.zeros((coords_ca_1.shape[0], coords_ca_2.shape[0], coords_ca_1.shape[1], coords_ca_1.shape[1]))
            for i in range(coords_ca_1.shape[0]):
                space_xs = coords_ca_1[i]
                mus = ot.unif(space_xs.shape[0])
                print(f"Conformer {i}: coords shape={space_xs.shape}")
                for j in range(coords_ca_2.shape[0]):
                    if i < j:
                        if symmetric: continue
                    space_ys = coords_ca_2[j]
                    nus = ot.unif(space_ys.shape[0])
                    if flip:
                        space_ys = space_ys.copy()
                        space_ys[:,0] *= -1
                    try:
                        loss_upper, plan, gap, loss_lower, loss_constant = gw_m_convex(mus, space_xs, nus, space_ys, {}, cost='CGW', gap_tol=1e-15, iter_max=iter_max, t=t, p_plus_implementation=config.p_plus_implementation, p_minus_implementation=config.p_minus_implementation, p_minus_dual_implementation=config.p_minus_dual_implementation, FW_iter=500)
                    except Exception as e:
                        print(f"Error comparing conformer {i} vs {j}: {e}")
                        loss_lower, gap, loss_upper, loss_constant = np.nan, np.nan, np.nan, np.nan
                    losses_lower_bound[i, j] = loss_lower
                    losses_upper_bound[i, j] = loss_upper
                    losses_gap[i, j] = gap
                    losses_constant[i, j] = loss_constant
                    plans[i, j] = plan
                    if symmetric:
                        losses_lower_bound[j, i] = losses_lower_bound[i, j]
                        losses_upper_bound[j, i] = losses_upper_bound[i, j]
                        losses_gap[j, i] = losses_gap[i, j]
                        losses_constant[j, i] = losses_constant[i, j]
                        plans[j, i] = plans[i, j]


                    print(f"Conformers {i} vs {j}: loss_lower={loss_lower:.8f}, loss_upper={loss_upper:.8f}, gap={gap:.8f}, constant={loss_constant:.8f}")
            return losses_lower_bound, losses_upper_bound, losses_gap, losses_constant, plans

        iter_max = config.max_iter
        n_proteins = config.n_models
        print("Cross CGW:")
        lower_bounds_cross, upper_bounds_cross, losses_gap_cross, losses_constant_cross, plans_cross = gw_m_convex_wrapper(coords_ca_1[:n_proteins], coords_ca_2[:n_proteins], t, symmetric=False, iter_max=iter_max, flip=flip)
        print("Self CGW 1:")
        lower_bounds_1, upper_bounds_1, losses_gap_1, losses_constant_1, plans_1 = gw_m_convex_wrapper(coords_ca_1[:n_proteins], coords_ca_1[:n_proteins], t, symmetric=True, iter_max=iter_max, flip=flip)
        print("Self CGW 2:")
        lower_bounds_2, upper_bounds_2, losses_gap_2, losses_constant_2, plans_2 = gw_m_convex_wrapper(coords_ca_2[:n_proteins], coords_ca_2[:n_proteins], t, symmetric=True, iter_max=iter_max, flip=flip)
        np.savez(config.xgw_plot_fname.replace('.png', f'_data_{config.unique_label}.npz'), 
                 config=OmegaConf.to_container(config, resolve=True, structured_config_mode=False),
                 lower_bounds_cross=lower_bounds_cross, 
                 upper_bounds_cross=upper_bounds_cross, 
                 lower_bounds_1=lower_bounds_1, 
                 upper_bounds_1=upper_bounds_1, 
                 lower_bounds_2=lower_bounds_2, 
                 upper_bounds_2=upper_bounds_2,
                 losses_gap_cross=losses_gap_cross,
                 losses_constant_cross=losses_constant_cross,
                 losses_gap_1=losses_gap_1,
                 losses_constant_1=losses_constant_1,
                 losses_gap_2=losses_gap_2,
                 losses_constant_2=losses_constant_2,
                 plans_cross=plans_cross,
                 plans_1=plans_1,
                 plans_2=plans_2,)

        fig = plt.figure(figsize=(12, 12))  
        plt.boxplot([safe_plot(upper_bounds_cross.flatten()), safe_plot(upper_bounds_1.flatten()), safe_plot(upper_bounds_2.flatten())])
        # label
        plt.xticks([1, 2, 3], ['cross', f'self_{label_1}', f'self_{label_2}'])
        # plt.yscale('log')
        plt.ylabel('CGW loss')
        plt.title(f'max_iter={iter_max} \n t={t:.4f}')
        plt.savefig(config.xgw_plot_fname, dpi=dpi)
        plt.close(fig)

        fig = plt.figure(figsize=(12, 12))  
        plt.boxplot([safe_plot(losses_constant_cross.flatten()), safe_plot(losses_constant_1.flatten()), safe_plot(losses_constant_2.flatten())])
        # label
        plt.xticks([1, 2, 3], ['cross', f'self_{label_1}', f'self_{label_2}'])
        # plt.yscale('log')
        plt.ylabel('Constant CGW loss')
        plt.title(f'max_iter={iter_max} \n t={t:.4f}')
        plt.savefig(config.xgw_plot_fname.replace('.png', f'_constant.png'), dpi=dpi)
        plt.close(fig)

if __name__ == "__main__":
    main()