import gemmi
import numpy as np
import matplotlib.pyplot as plt
import ot
import logging

from xgw.gromov_wasserstein_m_dist import gw_m_convex

# suppress noisy loggers; adjust level to WARNING or ERROR as needed
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("xgw").setLevel(logging.WARNING)
logging.getLogger("ot").setLevel(logging.WARNING)
logging.getLogger("gemmi").setLevel(logging.WARNING)

# module logger for informational messages in this script
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Set to INFO or DEBUG for more verbose output


def extract_coords_and_atoms(fname):
    st = gemmi.read_structure(fname)
    model_coords = []
    model_atoms = []
    for idx_model in range(len(st)):
        coords = []
        atoms = []
        for chain in st[idx_model]:
            for res in chain:
                for atom in res:
                    coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                    atoms.append((chain.name, res.seqid.num, atom.name))

        model_coords.append(np.array(coords))   # shape (N, 3)
        model_atoms.append(np.array(atoms))

    model_coords = np.array(model_coords)  # shape (M, N, 3)
    model_atoms = np.array(model_atoms)

    return model_coords, model_atoms

def selection_atoms(selection, model_coords, model_atoms):
    atom_type_idx = 2
    idx = (model_atoms[:,:,atom_type_idx] == selection)
    coords = model_coords[idx]
    return coords.reshape(model_coords.shape[0], -1, 3)

def gw_matrix(coords_ca_1, coords_ca_2, symmetric, n_skip):

    gw_losses = np.zeros((coords_ca_1.shape[0], coords_ca_2.shape[0]))
    
    for idx_1 in range(coords_ca_1.shape[0]):
        for idx_2 in range(coords_ca_2.shape[0]):
            if idx_1 < idx_2:
                if symmetric: continue
            C1 = ot.dist(coords_ca_1[idx_1,::n_skip], coords_ca_1[idx_1,::n_skip])
            C2 = ot.dist(coords_ca_2[idx_2,::n_skip], coords_ca_2[idx_2,::n_skip])
            p = ot.unif(coords_ca_1[idx_1,::n_skip].shape[0])
            q = ot.unif(coords_ca_2[idx_2,::n_skip].shape[0])
            gw_loss = ot.gromov_wasserstein2(C1, C2, p, q)
            gw_losses[idx_1, idx_2] = gw_loss
            if symmetric:
                gw_losses[idx_2, idx_1] = gw_loss
    return gw_losses

from dataclasses import dataclass
@dataclass
class Config:
    fname_1: str = '/Users/gw/repos/xgw/experiments/2M3T.cif'
    fname_2: str = '/Users/gw/repos/xgw/experiments/2M3U.cif'
    gw_plot_fname: str = '/Users/gw/repos/xgw/experiments/gw_loss_plot.png'
    selection: str = 'CA'
    n_models: int | None = None
    n_skip_every: int = 1
    compute_gw: bool = False
    compute_xgw: bool = True
    xgw_plot_fname: str = '/Users/gw/repos/xgw/experiments/xgw_loss_plot.png'


def main():
    import os
    config = Config()
    model_coords_1, model_atoms_1 = extract_coords_and_atoms(config.fname_1)
    label_1 = os.path.basename(config.fname_1).replace('.cif', '')
    np.savez(config.fname_1.replace('.cif', '_models.npz'), model_coords=model_coords_1, model_atoms=model_atoms_1)

    model_coords_2, model_atoms_2 = extract_coords_and_atoms(config.fname_2)
    label_2 = os.path.basename(config.fname_2).replace('.cif', '')
    np.savez(config.fname_2.replace('.cif', '_models.npz'), model_coords=model_coords_2, model_atoms=model_atoms_2)

    n_models = config.n_models
    dimension = 2
    coords_ca_1 = selection_atoms(config.selection, model_coords_1[:n_models], model_atoms_1[:n_models])[:,:,:dimension]
    coords_ca_2 = selection_atoms(config.selection, model_coords_2[:n_models], model_atoms_2[:n_models])[:,:,:dimension]

    coords_ca_1 = coords_ca_1 - np.mean(coords_ca_1, axis=1, keepdims=True)
    coords_ca_2 = coords_ca_2 - np.mean(coords_ca_2, axis=1, keepdims=True)
    r2_max = max(np.max(np.sum(coords_ca_1**2, axis=-1)), np.max(np.sum(coords_ca_2**2, axis=-1)))
    coords_ca_1 = coords_ca_1 / np.sqrt(r2_max)
    coords_ca_2 = coords_ca_2 / np.sqrt(r2_max)

    n_skip_every = config.n_skip_every

    if config.compute_gw:
        cross_gw_losses = gw_matrix(coords_ca_1, coords_ca_2, symmetric=False, n_skip=n_skip_every)
        self_gw_losses_1 = gw_matrix(coords_ca_1, coords_ca_1, symmetric=True, n_skip=n_skip_every)
        self_gw_losses_2 = gw_matrix(coords_ca_2, coords_ca_2, symmetric=True, n_skip=n_skip_every)

        fig = plt.figure(figsize=(12, 12))  
        plt.boxplot([cross_gw_losses.flatten(), self_gw_losses_1.flatten(), self_gw_losses_2.flatten()])
        plt.xticks([1, 2, 3], ['cross', f'self_{label_1}', f'self_{label_2}'])
        # plt.yscale('log')
        plt.ylabel('GW loss')
        plt.savefig(config.gw_plot_fname)
        plt.close(fig)

    if config.compute_xgw:
        r2_x = np.linalg.norm(coords_ca_1, axis=(-1)).max()
        r2_y = np.linalg.norm(coords_ca_2, axis=(-1)).max()
        r2_max = max(r2_x, r2_y)
        t = 24*r2_max / (2 + 24*r2_max)
        logger.info(f"Using t={t:.4f} based on r2_max={r2_max:.4f}")

        def gw_m_convex_wrapper(coords_ca_1, coords_ca_2, t, symmetric, iter_max):
            losses_lower_bound = np.zeros((len(coords_ca_1), len(coords_ca_2)), dtype=float)
            losses_upper_bound = np.zeros_like(losses_lower_bound)
            for i in range(coords_ca_1.shape[0]):
                space_xs = coords_ca_1[i]
                mus = ot.unif(space_xs.shape[0])
                logger.info(f"Conformer {i}: coords shape={space_xs.shape}")
                for j in range(coords_ca_2.shape[0]):
                    if i < j:
                        if symmetric: continue
                    space_ys = coords_ca_2[j]
                    nus = ot.unif(space_ys.shape[0])
                    try:
                        loss_lower, plan, c, loss_upper = gw_m_convex(mus, space_xs, nus, space_ys, {}, cost='CGW', cost_tol=1e-15, iter_max=iter_max, t=t)
                    except Exception as e:
                        logger.error(f"Error comparing conformer {i} vs {j}: {e}")
                        loss_lower, c, loss_upper = np.nan, np.nan, np.nan
                    losses_lower_bound[i, j] = loss_lower
                    losses_upper_bound[i, j] = loss_upper
                    if symmetric:
                        losses_lower_bound[j, i] = loss_lower
                        losses_upper_bound[j, i] = loss_upper


                    logger.info(f"Conformers {i} vs {j}: loss_lower={loss_lower:.4f}, loss_upper={loss_upper:.4f}, c={c:.4f}")
            return losses_lower_bound, losses_upper_bound

        iter_max = 2
        n_proteins = 2
        lower_bounds_cross, upper_bounds_cross = gw_m_convex_wrapper(coords_ca_1[:n_proteins], coords_ca_2[:n_proteins], t, symmetric=False, iter_max=iter_max)
        lower_bounds_1, upper_bounds_1 = gw_m_convex_wrapper(coords_ca_1[:n_proteins], coords_ca_1[:n_proteins], t, symmetric=True, iter_max=iter_max)
        lower_bounds_2, upper_bounds_2 = gw_m_convex_wrapper(coords_ca_2[:n_proteins], coords_ca_2[:n_proteins], t, symmetric=True, iter_max=iter_max)

        np.savez(config.fname_1.replace('.cif', f'_xgw_bounds_iter{iter_max}.npz'), 
                 config=config,
                 lower_bounds_cross=lower_bounds_cross, 
                 upper_bounds_cross=upper_bounds_cross, 
                 lower_bounds_1=lower_bounds_1, 
                 upper_bounds_1=upper_bounds_1, 
                 lower_bounds_2=lower_bounds_2, 
                 upper_bounds_2=upper_bounds_2)

        fig = plt.figure(figsize=(12, 12))  
        plt.boxplot([lower_bounds_cross.flatten(), lower_bounds_1.flatten(), lower_bounds_2.flatten()])
        # label
        plt.xticks([1, 2, 3], ['cross', f'self_{label_1}', f'self_{label_2}'])
        # plt.yscale('log')
        plt.ylabel('CGW loss')
        plt.title(f'max_iter={iter_max} \n t={t:.4f}')
        plt.savefig(config.xgw_plot_fname)
        plt.close(fig)

if __name__ == "__main__":
    main()