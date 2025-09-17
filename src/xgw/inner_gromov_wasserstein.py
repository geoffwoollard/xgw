import ot
import numpy as np

def double_inner(marginal, space):
    return np.einsum('i,j,i,j->', space**2, space**2, marginal, marginal)

def compute_gamma(marginal_a, marginal_b, space_x, space_y):
    return double_inner(marginal_a, space_x) + double_inner(marginal_b, space_y)

def covariance(space_x, space_y, pi):
    return np.einsum('i,ij,j->', space_x, pi, space_y)


def cost_function(space_x, space_y, sigma_pi_n):
    return sigma_pi_n * space_x * space_y

def line_search(gamma, space_x, space_y, pi_n, pi_n_1):
    alpha = np.einsum('i,j,k,l,ik,jl->', space_x, space_x, space_y, space_y, pi_n_1, pi_n_1)
    beta = np.einsum('i,j,k,l,ik,jl->', space_x, space_x, space_y, space_y, pi_n_1, pi_n)
    condition = 2*beta - alpha - gamma
    if condition > 0:
        tau = min(1, max(0,(beta - gamma)/ condition))
    elif alpha > gamma:
        tau = 1
    else:
        tau = 0
    gamma = alpha
    return tau, gamma


def igw_objective(space_x, space_y, pi):
    marginal_a = pi.sum(1)
    marginal_b = pi.sum(0)
    diag = double_inner(marginal_a, space_x) + double_inner(marginal_b, space_y) 
    cross = -2*np.einsum('i,j,k,l,ik,jl->', space_x, space_x, space_y, space_y, pi, pi)
    return diag + cross



def igw_algorithm1_1d():
    mu_a1, mu_a2, sigma_a = -0.5, 1, 0.1
    mu_b, sigma_b = 0, 0.2
    n_grid_1d_x = 100
    n_grid_1d_y = 50
    space_x = np.linspace(-1.5, 1.5, n_grid_1d_x)
    space_y = np.linspace(-1.5, 1.5, n_grid_1d_y)
    # create two Gaussian distributions
    marginal_a = np.abs(mu_a2)*np.exp(-0.5 * ((space_x - mu_a1) / sigma_a) ** 2)
    marginal_a += np.abs(mu_a1/mu_a2)*np.exp(-0.5 * ((space_x - mu_a2) / sigma_a) ** 2)
    marginal_b = np.exp(-0.5 * ((space_y - mu_b) / sigma_b) ** 2)
    marginal_a /= marginal_a.sum()
    marginal_b /= marginal_b.sum()

    gamma = compute_gamma(marginal_a, marginal_b, space_x, space_y)

    n_iters = 10
    pi_n = np.outer(marginal_a, marginal_b)

    for it in range(n_iters):
        sigma_pi_n = covariance(space_x, space_y, pi_n)
        cost = cost_function(space_x.reshape(-1,1), space_y.reshape(1,-1), sigma_pi_n)
        pi_n_1_hat = ot.emd(marginal_a, marginal_b, -cost)
        tau, gamma = line_search(gamma, space_x, space_y, pi_n, pi_n_1_hat)
        pi_n_1 = tau*pi_n_1_hat + (1-tau)*pi_n
        pi_n = pi_n_1
        print(f"it {it}: tau={tau}, igw={igw_objective(space_x, space_y, pi_n)}")


def igw_objective_d(space_x, space_y, pi):

    # Gram matrices
    Gx = space_x @ space_x.T        # shape (n1, n2)
    Gy = space_y @ space_y.T        # shape (m1, m2)

    # row/col sums
    r = pi.sum(axis=1)    # shape (n1,)
    c = pi.sum(axis=0)    # shape (m1,)

    # Term 1: sum_{i,j} Gx[i,j]^2 * r[i] * r[j]
    term1 = (r[:,None] * (Gx**2) * r[None,:]).sum()

    # Term 2: sum_{k,l} Gy[k,l]^2 * c[k] * c[l]
    term2 = (c[:,None] * (Gy**2) * c[None,:]).sum()

    # Term 3: -2 * tr(Pi.T @ Gx @ Pi @ Gy.T)
    term3 = -2 * np.trace(pi.T @ Gx @ pi @ Gy.T)

    return term1 + term2 + term3


def double_inner_vectorized_d(marginal, space):
    dot_sq = np.dot(space, space.T) ** 2  # shape (n, n)
    outer_marg = np.outer(marginal, marginal)  # shape (n, n)
    return np.sum(dot_sq * outer_marg)


def compute_gamma_d(marginal_a, marginal_b, space_x, space_y):
    return double_inner_vectorized_d(marginal_a, space_x) + double_inner_vectorized_d(marginal_b, space_y)


def covariance_vectorized_d(space_x, space_y, pi):
    return np.einsum('kd,lD,kl->dD', space_x, space_y, pi)


def covariance_d(space_x, space_y, pi):
    sigma_pi = np.zeros((len(space_x.T), len(space_y.T)))
    for i in range(len(space_x.T)):
        for j in range(len(space_y.T)):
            sigma_pi[i,j] = np.einsum('k,l,kl->', space_x[:,i], space_y[:,j], pi)
    return sigma_pi


def cost_function_d(space_x, space_y, sigma_pi_n): 
    return (space_x @ sigma_pi_n.T).dot(space_y.T)


def double_inner_with_two_joints(Gramm_x, Gramm_y, pi_1, pi_2):
    return np.trace(pi_1.T @ Gramm_x @ pi_2 @ Gramm_y.T)


def line_search_d(gamma, Gramm_x, Gramm_y, pi_n, pi_n_1):
    alpha = double_inner_with_two_joints(Gramm_x, Gramm_y, pi_n_1, pi_n_1)
    beta = double_inner_with_two_joints(Gramm_x, Gramm_y, pi_n, pi_n_1)
    condition = 2*beta - alpha - gamma
    if condition > 0:
        tau = min(1, max(0,(beta - gamma)/ condition))
    elif alpha > gamma:
        tau = 1
    else:
        tau = 0
    gamma = alpha
    return tau, gamma


def igw_algorithm1_2d():
    mu_a, sigma_a = np.array([-0.5, 0]), 0.3
    mu_b, sigma_b = np.array([0.5, 0]), 0.2
    n_grid_1d_x = 50
    n_grid_1d_y = 25
    def make_space_2d(n_grid):
        lin = np.linspace(-1, 1, n_grid)
        xx, yy = np.meshgrid(lin, lin)
        return np.vstack([xx.ravel(), yy.ravel()]).T
    space_x = make_space_2d(n_grid_1d_x)
    space_y = make_space_2d(n_grid_1d_y)

    marginal_a = np.exp(-0.5 * (((space_x - mu_a) / sigma_a) ** 2).sum(-1))
    marginal_b = np.exp(-0.5 * (((space_y - mu_b) / sigma_b) ** 2).sum(-1))
    marginal_a /= marginal_a.sum()
    marginal_b /= marginal_b.sum()

    gamma = compute_gamma_d(marginal_a, marginal_b, space_x, space_y)
    Gramm_x = space_x @ space_x.T        # (n_grid_1d_x, n_grid_1d_x)
    Gramm_y = space_y @ space_y.T        # (n_grid_1d_y, n_grid_1d_y)

    n_iters = 50
    pi_n = np.outer(marginal_a, marginal_b)
    print(f"igw={igw_objective_d(space_x, space_y, pi_n)}", pi_n.sum())

    for it in range(n_iters):
        sigma_pi_n = covariance_vectorized_d(space_x, space_y, pi_n)
        cost = cost_function_d(space_x, space_y, sigma_pi_n)
        pi_n_1_hat = ot.emd(marginal_a, marginal_b, -cost)
        tau, gamma = line_search_d(gamma, Gramm_x, Gramm_y, pi_n, pi_n_1_hat)
        pi_n_1 = tau*pi_n_1_hat + (1-tau)*pi_n
        pi_n = pi_n_1
        
        print(f"it {it}: tau={tau}, igw={igw_objective_d(space_x, space_y, pi_n)}")

    np.save("igw_2d_transport.npy", pi_n)


if __name__ == "__main__":
    igw_algorithm1_2d()