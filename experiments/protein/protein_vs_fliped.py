import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image as PILImage
import io
from mpl_toolkits.mplot3d import Axes3D

from proteins import extract_coords_and_atoms, selection_atoms
from xgw.gromov_wasserstein_m_dist import gw_m_convex

def gif(coords_ca_rotated, points_flipped, perm, r2_max, path):

    # Create frames
    frames = []
    n_frames = 50
    t_values = np.linspace(0, 1, n_frames)
    n_points = len(coords_ca_rotated)
    colors_idx = (np.arange(n_points) / n_points) % 1.0

    # Need full 3D coordinates, rescale back
    coords_ca_3d = coords_ca_rotated * np.sqrt(r2_max)
    points_flipped_3d = points_flipped[perm] * np.sqrt(r2_max)

    # Pre-compute axis limits (CRITICAL for stable plot)
    all_points = np.vstack([coords_ca_3d, points_flipped_3d])
    x_min, x_max = all_points[:, 0].min() - 1, all_points[:, 0].max() + 1
    y_min, y_max = all_points[:, 1].min() - 1, all_points[:, 1].max() + 1
    z_min, z_max = all_points[:, 2].min() - 1, all_points[:, 2].max() + 1

    for t in t_values:
        points_interpolated = (1-t)*coords_ca_3d + t*points_flipped_3d
        
        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        _ = ax.scatter(points_interpolated[:, 0], 
                            points_interpolated[:, 1],
                            points_interpolated[:, 2],
                            c=colors_idx,
                            cmap='hsv',
                            s=100, 
                            alpha=0.7,
                            edgecolors='white',
                            linewidth=0.5)
        
        # Connect points with lines
        ax.plot(points_interpolated[:, 0], 
            points_interpolated[:, 1],
            points_interpolated[:, 2],
            'k-', alpha=0.2, linewidth=1)
        ax.grid(False)
        
        # SET FIXED LIMITS (prevents rescaling)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        
        # SET FIXED VIEW ANGLE (prevents camera movement)
        ax.view_init(elev=20, azim=45)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Protein Interpolation (t={t:.2f})')
        
        # Disable automatic aspect ratio adjustment
        ax.set_box_aspect([1,1,1])
        
        # Save to buffer and convert with PIL
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img = PILImage.open(buf)
        frames.append(np.array(img))
        plt.close(fig)

    # Save GIF
    imageio.mimsave(path, frames, fps=20)
    print(f"3D GIF saved as '{path}'")

def optimal_rotation(points_source, points_target):
    """
    Find optimal rotation matrix R that minimizes ||R @ points_source.T - points_target.T||_F
    
    Uses Kabsch algorithm (SVD-based).
    
    Args:
        points_source: (N, 3) array
        points_target: (N, 3) array
    
    Returns:
        R: (3, 3) rotation matrix
        points_rotated: (N, 3) rotated source points
        rmse: root mean squared error after rotation
    """
    rmse_i = np.linalg.norm(points_source - points_target) / np.sqrt(len(points_source))


    # Center both point clouds
    centroid_source = points_source.mean(axis=0)
    centroid_target = points_target.mean(axis=0)
    
    source_centered = points_source - centroid_source
    target_centered = points_target - centroid_target
    
    # Compute covariance matrix
    H = source_centered.T @ target_centered  # (3, 3)
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Optimal rotation
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det=1, not reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply rotation
    points_rotated = (source_centered @ R.T) + centroid_target
    
    # RMSE
    rmse_f = np.linalg.norm(points_rotated - points_target) / np.sqrt(len(points_source))
    
    return R, points_rotated, rmse_i, rmse_f, centroid_target

def experiment(fname, cost, iter_max=20):
    
    model_coords, model_atoms = extract_coords_and_atoms(fname)
    coords_ca = selection_atoms(['CA'], model_coords, model_atoms)
    com = np.mean(coords_ca, axis=1, keepdims=True)

    coords_ca = coords_ca - com
    r2_max = np.max(np.sum(coords_ca**2, axis=-1))
    coords_ca = coords_ca[0] / np.sqrt(r2_max)

    mu = np.ones(len(coords_ca)) / len(coords_ca)
    points = coords_ca
    
    r2_max_after = np.max(np.sum(coords_ca**2, axis=-1))
    t = 8*r2_max_after / (2 + 8*r2_max_after)
    points_flipped = points.copy()
    points_flipped[:,0] *= -1
    _, plan, _, _, _ = gw_m_convex(mu, points, mu, points_flipped, {'numItermax': 10**10}, 
                                       cost=cost, gap_tol=1e-15, iter_max=iter_max, t=t, FW_iter=100, 
                                p_plus_implementation='h_to_v_popcount_sparse', 
                                p_minus_implementation='v_to_h_dual',
                                p_minus_dual_implementation='h_to_v_popcount_sparse') 
    perm = np.argmax(plan, axis=1)
    # Usage:
    R, coords_ca_rotated, rmse_i, rmse_f, centroid_target = optimal_rotation(points, points_flipped[perm])
    print(f"Rotation matrix:\n{R}")
    print(f"RMSE before {rmse_i:.6f} and after {rmse_f:.6f} alignment")



    path = f'protein_barycenter_{cost}.gif'
    gif(coords_ca_rotated, points_flipped, perm, r2_max, path)

if __name__ == "__main__":
    experiment('3PG0.cif', 'CGW')
    experiment('3PG0.cif', 'IGW')
