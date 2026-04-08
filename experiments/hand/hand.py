import cv2 as cv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from xgw.gromov_wasserstein_m_dist import gw_m_convex, classical_gw

def open_and_convert_to_grayscale(path):
    '''open and convert to grayscale'''
    img = Image.open(path).convert('L')
    img_array = np.array(img)
    return img_array

def filled_hand_experiment(path):
    img_array = open_and_convert_to_grayscale(path)

    img_array.shape

    crop = img_array[20:-20, 250:]
    # binarize
    crop = (crop > 128).astype(float)
    plt.imshow(crop, cmap='gray')
    plt.axis('off')

    # downsample image
    ds_factor = 4
    iter_max = 10
    crop = crop[::ds_factor, ::ds_factor]
    plt.imshow(crop, cmap='gray')
    plt.axis('off')
    print(crop.shape)


    i_mesh, j_mesh = np.meshgrid(np.arange(crop.shape[1], dtype=float), np.arange(crop.shape[0], dtype=float), indexing='ij')
    points = np.vstack([i_mesh.flatten(), j_mesh.flatten()]).T
    points /= np.max(np.linalg.norm(points, axis=1, keepdims=True))


    mu = crop.flatten() / crop.sum()

    mu_flipped = crop[::-1].flatten() / crop.sum()

    T, plan, c, _, _ = gw_m_convex(mu, points, mu_flipped, points, {'numItermax': 10**9}, cost='CGW', gap_tol=1e-15, iter_max=iter_max, FW_iter=100, p_plus_implementation='cdd') 

    ### original
    # vertical gradient
    binary = crop.astype(bool)
    H, W = binary.shape
    grad = np.linspace(0, 1, H)[:, None]
    grad = np.repeat(grad, W, axis=1)

    # convert gradient to colors
    cmap = plt.cm.viridis
    colors = cmap(grad)[..., :3]

    # apply mask
    colored = np.ones((H, W, 3))  # white background
    colored[binary] = colors[binary]

    plt.imshow(colored)
    plt.axis("off")
    plt.title('Original Image with Gradient Colors')
    plt.savefig('filled_original.png', dpi=300, bbox_inches='tight')

    ### perfect flip
    # vertical gradient
    binary = crop[::-1].astype(bool)
    H, W = binary.shape
    grad = np.linspace(0, 1, H)[:, None]
    grad = np.repeat(grad, W, axis=1)

    # convert gradient to colors
    cmap = plt.cm.viridis
    colors = cmap(grad)[..., :3][::-1]

    # apply mask
    colored = np.ones((H, W, 3))  # white background
    colored[binary] = colors[binary]

    plt.imshow(colored)
    plt.title('Perfect Mirror Image')
    plt.axis("off")
    plt.savefig('filled_perfect_flip.png', dpi=300, bbox_inches='tight')

    ### CGW
    perm = plan.argmax(axis=0)
    # vertical gradient
    binary = crop[::-1].astype(bool)
    H, W = binary.shape
    grad = np.linspace(0, 1, H)[:, None]
    grad = np.repeat(grad, W, axis=1)

    # convert gradient to colors
    cmap = plt.cm.viridis
    colors = cmap(grad)[..., :3][::-1]

    # apply mask
    colored = np.ones((H, W, 3))  # white background
    transported_colors = colors.reshape(-1,3)[perm].reshape(colored.shape)
    colored[binary] = transported_colors[binary]

    plt.imshow(colored)
    plt.axis("off")
    plt.title('Chiral GW')
    plt.savefig('filled_cgw.png', dpi=300, bbox_inches='tight')

    ### classical gw
    _, plan_gw, _, _, _ = classical_gw(mu, points, mu_flipped, points, {'numItermax': 10**10},  cost_tol=1e-18, iter_max=iter_max, FW_iter=100, p_plus_implementation='cdd') 
    perm = plan_gw.argmax(axis=0)
    # vertical gradient
    binary = crop[::-1].astype(bool)
    H, W = binary.shape
    grad = np.linspace(0, 1, H)[:, None]
    grad = np.repeat(grad, W, axis=1)

    # convert gradient to colors
    cmap = plt.cm.viridis
    colors = cmap(grad)[..., :3][::-1]

    # apply mask
    colored = np.ones((H, W, 3))  # white background
    transported_colors = colors.reshape(-1,3)[perm].reshape(colored.shape)
    colored[binary] = transported_colors[binary]

    plt.imshow(colored)
    plt.axis("off")
    plt.title('Classical GW')
    plt.savefig('filled_classicalgw.png', dpi=300, bbox_inches='tight')

def plot(points, title, fname, colors_perm, figsize=(10, 8)):
    # Color points as rainbow on scatter plot with looping rainbow
    # Create looping rainbow colors (wraps around multiple times)

    n_points = len(points)
    n_loops = 1  # number of times to loop through the rainbow
    colors_idx = (np.arange(n_points) / n_points * n_loops) % 1.0  # normalized to [0, 1]
    colors_idx = colors_idx[colors_perm]
    # Plot points with looping rainbow colors
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(points[:, 0], points[:, 1], 
                        c=colors_idx,  # looping color index
                        cmap='hsv',    # hsv wraps nicely
                        s=100, 
                        alpha=0.7,
                        edgecolors='white',
                        linewidth=0.5)


    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, )

    ax.set_aspect('equal')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)

def border_hand_experiment(path):
    img_array = open_and_convert_to_grayscale(path)
    crop = img_array[20:-10, 250:]
    # binarize
    crop = (crop > 128).astype(float)
    ret, thresh = cv.threshold(crop*128, 127, 255, 0)
    thresh = thresh.astype(np.uint8)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)     
    outline = contours[0].reshape(-1, 2)
    
    import seaborn as sns

    # Set poster style (larger fonts, thicker lines)
    sns.set_context("poster")  # options: "paper", "notebook", "talk", "poster"
    sns.set_style("white")

    n_skip = 3
    points = outline.astype(float)
    points = points[::n_skip]
    points /= np.max(np.linalg.norm(points, axis=1, keepdims=True))
    points -= points.mean(axis=0, keepdims=True)
    points_flipped = points.copy()
    points_flipped[:,0] *= -1
    mu = np.ones(len(points)) / len(points)

    fname = 'hand_contour_reference.svg'
    title = 'Hand Contour - Reference'
    perm = np.arange(len(points))
    plot(points, title, fname, perm)

    _, plan_cgw, _, _, _ = gw_m_convex(mu, points, mu, points_flipped, {'numItermax': 10**9}, cost='CGW', gap_tol=1e-15, iter_max=300, FW_iter=100, 
                               p_plus_implementation='h_to_v_popcount_sparse', 
                               p_minus_implementation='v_to_h_dual',
                               p_minus_dual_implementation='h_to_v_popcount_sparse') 
    
    title = 'Hand Contour - Chiral GW'
    fname = 'hand_contour_cgw.svg'
    perm = plan_cgw.argmax(axis=0)
    plot(points, title, fname, perm)

    _, plan_classicalgw, _, _, _ = classical_gw(mu, points, mu, points_flipped, {'numItermax': 10**10}, cost_tol=1e-18, iter_max=100, FW_iter=100, 
                                p_plus_implementation='cdd', 
                                p_minus_implementation='cdd',
                                ) 

    title = 'Hand Contour - Classical GW'
    fname = 'hand_contour_classicalgw.svg'
    perm = plan_classicalgw.argmax(axis=0)
    plot(points, title, fname, perm)

    np.savez('hand_contour_plans.npz', plan_cgw=plan_cgw, plan_classicalgw=plan_classicalgw)
    assert np.allclose(plan_classicalgw * len(plan_classicalgw), np.eye(len(plan_classicalgw)))

if __name__ == '__main__':
    path = 'hand.png'
    filled_hand_experiment(path)