from xgw.gromov_wasserstein_m_dist import gw_m_convex, classical_gw

# open and convert to grayscale
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    fname = 'hand.png'
    img = Image.open(fname).convert('L')
    img_array = np.array(img)
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
    plt.savefig('hand_original.png', dpi=300, bbox_inches='tight')

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
    plt.savefig('hand_perfect_flip.png', dpi=300, bbox_inches='tight')

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
    plt.savefig('hand_cgw.png', dpi=300, bbox_inches='tight')

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
    plt.savefig('hand_gw.png', dpi=300, bbox_inches='tight')