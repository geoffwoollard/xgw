import ot
import numpy as np


def safe_plot(arr):
    return arr[np.isfinite(arr)]

def gw_matrix(coords_ca_1, coords_ca_2, symmetric, n_skip, flip):

    gw_losses = np.zeros((coords_ca_1.shape[0], coords_ca_2.shape[0]))
    
    for idx_1 in range(coords_ca_1.shape[0]):
        for idx_2 in range(coords_ca_2.shape[0]):
            if idx_1 < idx_2:
                if symmetric: continue
            if flip: 
                coords_ca_2_ = coords_ca_2[idx_2,::n_skip].copy()
                coords_ca_2_[:,0] *= -1
            else:
                coords_ca_2_ = coords_ca_2[idx_2,::n_skip]

            C1 = ot.dist(coords_ca_1[idx_1,::n_skip], coords_ca_1[idx_1,::n_skip])
            C2 = ot.dist(coords_ca_2_, coords_ca_2_)
            p = ot.unif(coords_ca_1[idx_1,::n_skip].shape[0])
            q = ot.unif(coords_ca_2_.shape[0])
            gw_loss = ot.gromov_wasserstein2(C1, C2, p, q)
            gw_losses[idx_1, idx_2] = gw_loss
            if symmetric:
                gw_losses[idx_2, idx_1] = gw_loss
    return gw_losses