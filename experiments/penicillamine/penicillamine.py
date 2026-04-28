from rdkit import Chem
from xgw.gromov_wasserstein_m_dist import gw_m_convex
import numpy as np
from matplotlib import pyplot as plt

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

def main(fname_molecule_input, fname_output,iter_max, n_conformers, noise_level):
    coords, _ = read_molecule(fname_molecule_input)
    points = center_and_normalize(coords)
    conformers, noise = make_conformers(points, n_conformers=n_conformers, noise_level=noise_level)
    r2_max_after = np.linalg.norm(conformers.reshape(-1, 3), axis=1).max()**2
    t = 8*r2_max_after / (2 + 8*r2_max_after) 

    mu = np.ones(len(points)) / len(points)

    cgws = np.zeros((n_conformers, n_conformers))
    rmsds = np.zeros((n_conformers, n_conformers))
    plans = np.zeros((n_conformers, n_conformers, len(points), len(points)))

    for i in range(n_conformers):
        for j in range(i, n_conformers):
            if i == 0 and j == 1:
                cgw, plan, _, _, _ = gw_m_convex(mu, conformers[i], mu, conformers[j], {'numItermax': 10**9}, cost='CGW', gap_tol=1e-15, iter_max=iter_max, t=t, FW_iter=100,
                                                p_plus_implementation='h_to_v_popcount_sparse', 
                                                p_minus_implementation='v_to_h_dual', 
                                                p_minus_dual_implementation='h_to_v_popcount_sparse') 
            else:
                cgw = np.nan
                plan = np.eye(len(points))
            cgws[i, j] = cgws[j, i] = cgw
            plans[i, j] = plans[j, i] = plan

            # Compute RMSD
            rmsd = np.linalg.norm(conformers[i] - conformers[j]) / np.sqrt(len(points))
            rmsds[i, j] = rmsds[j, i] = rmsd

    np.savez(fname_output, conformers=conformers, noise=noise, cgws=cgws, rmsds=rmsds, plans=plans)

def plot(fname):
    data = np.load(fname)
    cgws = data['cgws']
    rmsds = data['rmsds']

    plt.imshow(rmsds, cmap='gray')
    plt.title('RMSD Distances')
    plt.colorbar()
    plt.savefig('rmsd.png')
    plt.clf()

    plt.imshow(cgws, cmap='gray')
    plt.title('CGW Distances')
    plt.colorbar()
    plt.savefig('cgw.png')
    plt.clf()

    plt.scatter(cgws.flatten(), rmsds.flatten())
    plt.xlabel('CGW Distance')
    plt.ylabel('RMSD Distance')
    plt.title('CGW vs RMSD')
    plt.savefig('cgw_vs_rmsd.png')
    plt.clf()

    # mismatch from id
    plans = data['plans']
    traces = np.einsum('ijkk->ij', plans)
    plt.imshow(traces, cmap='gray')
    plt.title('Trace of Optimal Transport Plans')
    plt.colorbar()
    plt.savefig('trace.png')
    plt.clf()



if __name__ == "__main__":
    n_conformers = 3
    noise_level = 0.001
    iter_max = 40
    
    fname_molecule_input = "/home/gw/repos/xgw/experiments/penicillamine/Conformer3D_COMPOUND_CID_4727.sdf"
    fname = f"/home/gw/repos/xgw/experiments/penicillamine/penicillamine_conformers_nconfcormers{n_conformers}_noiselevel{noise_level}_itermax{iter_max}.npz"
    main(fname_molecule_input, fname, iter_max, n_conformers, noise_level )
    plot(fname)