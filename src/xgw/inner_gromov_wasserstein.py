import ot
import numpy as np

def double_inner(marginal, space):
    return np.einsum('i,j,i,j->', space**2, space**2, marginal, marginal)

def compute_b(marginal_a, marginal_b, space_x, space_y):
    return double_inner(marginal_a, space_x) + double_inner(marginal_b, space_y)

def covariance(space_x, space_y, pi):
    return np.einsum('i,ij,j->', space_x, pi, space_y)


def cost_function(space_x, space_y, sigma_pi_n):
    return sigma_pi_n * space_x * space_y

def line_search(b, space_x, space_y, pi_n, pi_n_1):
    alpha = -2*np.einsum('i,j,k,l,ik,jl->', space_x, space_x, space_y, space_y, pi_n_1, pi_n_1)
    beta = b - 4*np.einsum('i,j,k,l,ik,jl->', space_x, space_x, space_y, space_y, pi_n_1, pi_n)
    if alpha > 0:
        tau = min(1, max(0, -beta / 2*alpha))
    elif alpha + beta < 0:
        tau = 1
    else:
        tau = 0
    return tau

def igw_objective(space_x, space_y, pi):
    marginal_a = pi.sum(1)
    marginal_b = pi.sum(0)
    diag = double_inner(marginal_a, space_x) + double_inner(marginal_b, space_y) 
    cross = -2*np.einsum('i,j,k,l,ik,jl->', space_x, space_x, space_y, space_y, pi, pi)
    return diag + cross


def igw_algorithm1_1d():
    mu_a, sigma_a = -0.5, 0.1
    mu_b, sigma_b = 0.5, 0.2
    n_grid_1d_x = 100
    n_grid_1d_y = 50
    space_x = np.linspace(-1.5, 1.5, n_grid_1d_x)
    space_y = np.linspace(-1.5, 1.5, n_grid_1d_y)
    # create two Gaussian distributions
    marginal_a = np.exp(-0.5 * ((space_x - mu_a) / sigma_a) ** 2)
    marginal_b = np.exp(-0.5 * ((space_y - mu_b) / sigma_b) ** 2)
    marginal_a /= marginal_a.sum()
    marginal_b /= marginal_b.sum()

    b = compute_b(marginal_a, marginal_b, space_x, space_y)


    n_iters = 3
    pi_n = np.outer(marginal_a, marginal_b)

    for it in range(n_iters):
        sigma_pi_n = covariance(space_x, space_y, pi_n)
        cost = cost_function(space_x.reshape(-1,1), space_y.reshape(1,-1), sigma_pi_n)
        pi_n_1_hat = ot.emd(marginal_a, marginal_b, cost)
        tau = line_search(b, space_x, space_y, pi_n, pi_n_1_hat)
        pi_n_1 = tau*pi_n_1_hat + (1-tau)*pi_n
        pi_n = pi_n_1

        
        print(f"it {it}: tau={tau}, igw={igw_objective(space_x, space_y, pi_n)}")

if __name__ == "__main__":
    igw_algorithm1_1d()