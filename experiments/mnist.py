import numpy as np
import logging
import pandas as pd
from tqdm import tqdm
import time
from dataclasses import dataclass, asdict
import ot

from xgw.gromov_wasserstein_m_dist import gw_m_convex
from xgw.utils import gw_matrix

logging.disable(logging.CRITICAL)

def space_MNIST(mnist, idx):
    image = mnist['images'][idx]
    space_x = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
    space_x = np.vstack([space_x[0].ravel(), space_x[1].ravel()]).T
    mu = image.ravel()
    mu = mu / np.sum(mu)
    return mu, space_x

def center(mu, space_x):
    weighted_sum = np.sum(mu[:, np.newaxis] * space_x, axis=0)
    return weighted_sum / np.sum(mu)

def sparse_mnist(mnist, idx):
    mus, space_xs = space_MNIST(mnist, idx)
    mus_nonzero = mus[mus>0]
    space_xs = space_xs[mus>0]
    space_xs = space_xs - center(mus_nonzero, space_xs).reshape(1, -1)
    return mus_nonzero, space_xs

@dataclass
class Config:
    mnist_path: str = '/Users/gw/repos/xgw/experiments/mnist_100.npz'
    output_csv: str = '/Users/gw/repos/xgw/experiments/mnist_cgw_distances.csv'
    unique_label: str = time.strftime("%Y%m%d-%H%M%S")
    t_eps: float = 1e-2
    iter_max: int = 100
    n_digits: int = 100
    space_scale: float = 0.5
    cost_tol: float = 1e-4
    random_seed: int = 0
    compute_gw: bool = True
    compute_xgw: bool = False


def main():
    config = Config()
    mnist = np.load(config.mnist_path)
    # idx_1 = 98
    # idx_2 = 90
    # mu_1, space_xs_1 = sparse_mnist(mnist, idx_1)
    # mu_2, space_xs_2 = sparse_mnist(mnist, idx_2)

    # box_size = 28
    # r2_max = (box_size/2)**2
    t = 1/2 + config.t_eps
    # t = (t + 9) / 10
    print(f"t={t:.4f}")

    # loss_upper_11, plan_11, gap_11, loss_lower_11, loss_constant_11 = gw_m_convex(mu_1, space_xs_1, mu_1, space_xs_1, {}, cost='CGW', cost_tol=1e-15, iter_max=100, t=t)
    # loss_upper_22, plan_22, gap_22, loss_lower_22, loss_constant_22 = gw_m_convex(mu_2, space_xs_2, mu_2, space_xs_2, {}, cost='CGW', cost_tol=1e-15, iter_max=30, t=t)
    # loss_upper_12, plan_12, gap_12, loss_lower_12, loss_constant_12 = gw_m_convex(mu_1, space_xs_1, mu_2, space_xs_2, {}, cost='CGW', cost_tol=1e-15, iter_max=100, t=t)

    # print(f"Loss upper 11: {loss_upper_11:.6f}, gap 11: {gap_11:.6f}, loss lower 11: {loss_lower_11:.6f}, loss constant 11: {loss_constant_11:.6f}")
    # print(f"Loss upper 22: {loss_upper_22:.6f}, gap 22: {gap_22:.6f}, loss lower 22: {loss_lower_22:.6f}, loss constant 22: {loss_constant_22:.6f}")
    # print(f"Loss upper 12: {loss_upper_12:.6f}, gap 12: {gap_12:.6f}, loss lower 12: {loss_lower_12:.6f}, loss constant 12: {loss_constant_12:.6f}")

    # np.savez('mnist_cgw_results.npz',
    #          mu_1=mu_1, space_xs_1=space_xs_1, 
    #          mu_2=mu_2, space_xs_2=space_xs_2,
    #          loss_upper_11=loss_upper_11, plan_11=plan_11, gap_11=gap_11, loss_lower_11=loss_lower_11, loss_constant_11=loss_constant_11,      
    #          loss_upper_22=loss_upper_22, plan_22=plan_22, gap_22=gap_22, loss_lower_22=loss_lower_22, loss_constant_22=loss_constant_22,
    #          loss_upper_12=loss_upper_12, plan_12=plan_12, gap_12=gap_12, loss_lower_12=loss_lower_12, loss_constant_12=loss_constant_12)

        
    n_digits = config.n_digits
    n_total = len(mnist['images'])
    d_list = []
    iter_max = config.iter_max
    
    np.random.seed(config.random_seed)
    idxs = np.random.choice(n_total, n_digits, replace=False)
    for idx_1 in tqdm(idxs, desc="Computing CGW distances", total=n_digits):
        mu_1, space_xs_1 = sparse_mnist(mnist, idx_1)
        digit_1 = mnist['labels'][idx_1]
        for idx_2 in idxs:
            digit_2 = mnist['labels'][idx_2]
            if digit_1 < digit_2:
                pass
            else:
                for flip in [False, True]:
                    mu_2, space_xs_2 = sparse_mnist(mnist, idx_2)
                    if flip:
                        space_xs_2[:, 0] = -space_xs_2[:, 0]
                        print(f"Flipped digit {digit_2} (idx {idx_2}) horizontally")
                
                    dict_ij = {'idx_1': idx_1, 
                                    'idx_2': idx_2, 
                                    'flip': flip,
                                    'digit_1': digit_1,
                                    'digit_2': digit_2,
                                    'random_seed': config.random_seed,
                    }

                    if config.compute_gw:
                        C1 = ot.dist(space_xs_1, space_xs_1)
                        C2 = ot.dist(space_xs_2, space_xs_2)
                        gw_loss = ot.gromov_wasserstein2(C1, C2, mu_1, mu_2) 
                        dict_ij['gw_loss'] = gw_loss
                    if config.compute_xgw:
                        try:
                            loss_upper, _, gap_11, loss_lower, loss_constant = gw_m_convex(mu_1, 
                                                                                        config.space_scale*space_xs_1, 
                                                                                        mu_2, 
                                                                                        config.space_scale*space_xs_2, 
                                                                                        {}, 
                                                                                        cost='CGW', 
                                                                                        cost_tol=config.cost_tol, 
                                                                                        iter_max=config.iter_max, 
                                                                                        t=t
                                                                                        )
                            
                            dict_ij.update({'idx_1': idx_1, 
                                        'idx_2': idx_2, 
                                        'flip': flip,
                                        'digit_1': digit_1,
                                        'digit_2': digit_2,
                                        'loss_upper': loss_upper,
                                        'gap': gap_11,
                                        'loss_lower': loss_lower,
                                        'loss_constant': loss_constant,
                                        't': t,
                                        'iter_max': iter_max,
                                        'space_scale': config.space_scale,
                                        })
                            print(f"Computed CGW distance between idx {idx_1} (digit {digit_1}) and idx {idx_2} (digit {digit_2}), flip={flip}: loss_upper={loss_upper:.6f}, gap={gap_11:.6f}, loss_lower={loss_lower:.6f}, loss_constant={loss_constant:.6f}")
                        except Exception as e:
                            print(f"Failed for idx {idx_1} and idx {idx_2}, flip={flip}: {e}")
                    d_list.append(dict_ij)
 
    df = pd.DataFrame(d_list)
    df.to_csv(config.output_csv.replace('.csv', f'_{config.unique_label}.csv'), index=False)
    np.savez(config.output_csv.replace('.csv', f'_{config.unique_label}.npz'), config=asdict(config), data=df.to_dict(orient='list'))

if __name__ == "__main__":
    main()