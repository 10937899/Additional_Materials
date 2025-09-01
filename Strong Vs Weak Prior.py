import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import seaborn as sns

# define the SIR Model
def sir_model(t, y, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Solve the SIR equation
def solve_sir(beta, gamma, I0, t_eval, N):
    S0 = N - I0
    R0 = 0
    sol = solve_ivp(sir_model, [t_eval[0], t_eval[-1]], [S0, I0, R0],
                    args=(beta, gamma, N), t_eval=t_eval)
    return sol.y[1]

# Poisson Log-likelihhod
def poisson_log_likelihood(I_obs, I_model, scale=1000):
    I_model = np.maximum(I_model, 1e-6)
    return np.sum(I_obs * np.log(I_model) - I_model) / scale

# Flat Prior
def log_prior(beta, gamma):
    if beta <= 0 or gamma <= 0:
        return -np.inf
    return 0

# Jeffreys' Prior
def jeffreys_log_prior(beta, gamma, t_eval, I0, N,
                       eps=1e-5, reg=1e-12):
    """
    Jeffreys prior for Poisson with mean lambda_t = I(t; beta, gamma):
    log pi_J = 0.5 * log det( Sum_t (1/lambda_t) * grad lam_t * grad lam_t^T ).
    """
    if beta <= 0 or gamma <= 0:
        return -np.inf

    lam0 = solve_sir(beta, gamma, I0, t_eval, N)
    lam0 = np.maximum(lam0, 1e-12)

    # Finite-difference sensitivities
    lam_b = solve_sir(beta + eps, gamma, I0, t_eval, N)
    lam_g = solve_sir(beta, gamma + eps, I0, t_eval, N)
    dlam_db = (lam_b - lam0) / eps
    dlam_dg = (lam_g - lam0) / eps

    # Fisher information (2x2)
    w = 1.0 / lam0
    F11 = np.sum(w * dlam_db * dlam_db)
    F22 = np.sum(w * dlam_dg * dlam_dg)
    F12 = np.sum(w * dlam_db * dlam_dg)
    detF = F11 * F22 - F12 * F12

    # Regularize near-singular cases (flat ridge -> det ~ 0)
    detF = max(detF, reg)
    return 0.5 * np.log(detF)

# Log-posterior using Jeffreys Prior
def log_posterior_R0_gamma(phi, I_obs, t_eval, I0, N):
    R0, gamma = float(phi[0]), float(phi[1])
    if R0 <= 0 or gamma <= 0:
        return -np.inf
    beta = R0 * gamma

    I_model = solve_sir(beta, gamma, I0, t_eval, N)
    ll = poisson_log_likelihood(I_obs, I_model) 
    lj = jeffreys_log_prior(beta, gamma, t_eval, I0, N)
    return ll + lj + np.log(gamma)

# Batchwise Adaptive Metropolis for R0 and gamma
def batch_adaptive_mcmc_phi(I_obs, t_eval, I0, N, n_batches=3, iter_per_batch=2000, burn_ratio=0.5, phi0=None):
    dim = 2
    proposal_cov = 0.01 * np.eye(dim)
    chains = []
    
    if phi0 is None:
        phi = np.array([0.5/0.1, 0.1], dtype=float)  # (R0, γ)
    else:
        phi = np.array(phi0, dtype=float)
    epsilon = 1e-6

    log_post_curr = log_posterior_R0_gamma(phi, I_obs, t_eval, I0, N)

    for _ in range(n_batches):
        chain_phi = np.zeros((iter_per_batch, dim))
        chain_phi[0] = phi

        for t in range(1, iter_per_batch):
            proposal = np.random.multivariate_normal(chain_phi[t - 1], proposal_cov)
            log_post_prop = log_posterior_R0_gamma(proposal, I_obs, t_eval, I0, N)

            if np.log(np.random.rand()) < (log_post_prop - log_post_curr):
                chain_phi[t] = proposal
                log_post_curr = log_post_prop
            else:
                chain_phi[t] = chain_phi[t - 1]

        burn_in = int(iter_per_batch * burn_ratio)
        chain_post_burn = chain_phi[burn_in:]
        proposal_cov = np.cov(chain_post_burn.T) + epsilon * np.eye(dim)
        phi = chain_phi[-1]
        chains.append(chain_phi)

    # transform stacked phi-chain to (beta,gamma) 
    chain_phi_full = np.vstack(chains)
    R0_chain = chain_phi_full[:, 0]
    gamma_chain = chain_phi_full[:, 1]
    beta_chain = R0_chain * gamma_chain
    return np.column_stack([beta_chain, gamma_chain])

# Windows wrapper in phi
def batch_adaptive_for_windows_R0gamma(windows, I_obs, days, I0, N):
    results = {}
    for w in windows:
        t_eval = days[:w]
        I_obs_w = I_obs[:w]
        chain_bg = batch_adaptive_mcmc_phi(I_obs_w, t_eval, I0, N, n_batches=3, iter_per_batch=2000)
        # remove 1 batch of burnin
        beta_chain = chain_bg[2000:, 0]
        gamma_chain = chain_bg[2000:, 1]
        results[w] = {
            "beta": beta_chain,
            "gamma": gamma_chain,
            "beta_minus_gamma": beta_chain - gamma_chain
        }
    return results

# Batchwise phi version that returns per-batch chains in (beta,gamma) for  trace plots
def batch_adaptive_mcmc_all_chains_R0gamma(I_obs, t_eval, I0, N, n_batches=3, iter_per_batch=4000, burn_ratio=0.5, phi0=None):
    dim = 2
    proposal_cov = 0.0001 * np.eye(dim)
    all_chains_bg = []
    if phi0 is None:
        phi = np.array([0.5/0.1, 0.1], dtype=float)  
    else:
        phi = np.array(phi0, dtype=float)
    epsilon = 1e-6

    log_post_curr = log_posterior_R0_gamma(phi, I_obs, t_eval, I0, N)

    for _ in range(n_batches):
        chain_phi = np.zeros((iter_per_batch, dim))
        chain_phi[0] = phi

        for t in range(1, iter_per_batch):
            proposal = np.random.multivariate_normal(chain_phi[t - 1], proposal_cov)
            log_post_prop = log_posterior_R0_gamma(proposal, I_obs, t_eval, I0, N)
            if np.log(np.random.rand()) < (log_post_prop - log_post_curr):
                chain_phi[t] = proposal
                log_post_curr = log_post_prop
            else:
                chain_phi[t] = chain_phi[t - 1]

        burn_in = int(iter_per_batch * burn_ratio)
        chain_post_burn = chain_phi[burn_in:]
        proposal_cov = np.cov(chain_post_burn.T) + epsilon * np.eye(dim)
        phi = chain_phi[-1]

        # transform this batch to (beta,gamma)
        beta_batch = chain_phi[:, 0] * chain_phi[:, 1]
        gamma_batch = chain_phi[:, 1]
        all_chains_bg.append(np.column_stack([beta_batch, gamma_batch]))

    return all_chains_bg

# Strongly-informative prior on (R0, gamma), evaluated in (beta, gamma) coords 
def log_prior_strong(beta, gamma, R0_ref=None, gamma_ref=None, s_R0=0.15, s_g=0.15):
    if beta <= 0 or gamma <= 0:
        return -np.inf
    if R0_ref is None or gamma_ref is None:
        # Fallback to values in your script if present
        # (Assumes you defined beta_true, gamma_true)
        R0_ref = beta_true / gamma_true
        gamma_ref = gamma_true
    R0 = beta / gamma
    # LogNormal log-density (includes Jacobian term -log(x))
    lp_R0 = -0.5 * ((np.log(R0) - np.log(R0_ref)) / s_R0) ** 2 - np.log(R0 * s_R0 * np.sqrt(2*np.pi))
    lp_g  = -0.5 * ((np.log(gamma) - np.log(gamma_ref)) / s_g) ** 2 - np.log(gamma * s_g * np.sqrt(2*np.pi))
    return lp_R0 + lp_g

# Log-posterior with STRONG prior (no Jeffreys) 
def log_posterior_strong(theta, I_obs, t_eval, I0, N):
    beta, gamma = theta
    if beta <= 0 or gamma <= 0:
        return -np.inf
    I_model = solve_sir(beta, gamma, I0, t_eval, N)
    return poisson_log_likelihood(I_obs, I_model) + log_prior_strong(beta, gamma)

# Batchwise Adaptive MCMC (STRONG prior) 
def batch_adaptive_mcmc_strong(I_obs, t_eval, I0, N, n_batches=3, iter_per_batch=2000, burn_ratio=0.5):
    dim = 2
    proposal_cov = 0.01 * np.eye(dim)
    chains = []
    theta = np.array([0.5, 0.1])  # keep same start for fair comparison
    epsilon = 1e-6

    log_post_curr = log_posterior_strong(theta, I_obs, t_eval, I0, N)

    for _ in range(n_batches):
        chain = np.zeros((iter_per_batch, dim))
        chain[0] = theta

        for t in range(1, iter_per_batch):
            proposal = np.random.multivariate_normal(chain[t - 1], proposal_cov)
            log_post_prop = log_posterior_strong(proposal, I_obs, t_eval, I0, N)

            if np.log(np.random.rand()) < (log_post_prop - log_post_curr):
                chain[t] = proposal
                log_post_curr = log_post_prop
            else:
                chain[t] = chain[t - 1]

        burn_in = int(iter_per_batch * burn_ratio)
        chain_post_burn = chain[burn_in:]
        proposal_cov = np.cov(chain_post_burn.T) + epsilon * np.eye(dim)
        theta = chain[-1]
        chains.append(chain)

    return np.vstack(chains)

# Run strong-prior AM across windows; return dict matching the current structure
def batch_adaptive_for_windows_strong(windows, I_obs, days, I0, N):
    results = {}
    for w in windows:
        t_eval = days[:w]
        I_obs_w = I_obs[:w]
        chain = batch_adaptive_mcmc_strong(I_obs_w, t_eval, I0, N, n_batches=3, iter_per_batch=2000)
        beta_chain = chain[2000:, 0]  # drop 1 batch, same as your Jeffreys path
        gamma_chain = chain[2000:, 1]
        results[w] = {
            "beta": beta_chain,
            "gamma": gamma_chain,
            "beta_minus_gamma": beta_chain - gamma_chain
        }
    return results


# Per-batch chains (for trace plots per batch), gamma-only prior version 
def batch_adaptive_mcmc_all_chains_strong_gamma(I_obs, t_eval, I0, N,
                                                n_batches=3, iter_per_batch=4000, burn_ratio=0.5):
    dim = 2
    proposal_cov = 0.0001 * np.eye(dim)   # match your per-batch trace function
    all_chains = []
    theta = np.array([0.5, 0.1])         # same as your earlier per-batch starter
    epsilon = 1e-6

    log_post_curr = log_posterior_strong(theta, I_obs, t_eval, I0, N)

    for _ in range(n_batches):
        chain = np.zeros((iter_per_batch, dim))
        chain[0] = theta

        for t in range(1, iter_per_batch):
            proposal = np.random.multivariate_normal(chain[t - 1], proposal_cov)
            log_post_prop = log_posterior_strong(proposal, I_obs, t_eval, I0, N)
            if np.log(np.random.rand()) < (log_post_prop - log_post_curr):
                chain[t] = proposal
                log_post_curr = log_post_prop
            else:
                chain[t] = chain[t - 1]

        burn_in = int(iter_per_batch * burn_ratio)
        chain_post_burn = chain[burn_in:]
        proposal_cov = np.cov(chain_post_burn.T) + epsilon * np.eye(dim)
        theta = chain[-1]
        all_chains.append(chain)

    return all_chains


# Simulate full epidemic 
N = 1_000_000
beta_true = 0.6
gamma_true = 0.1
R0_true = beta_true / gamma_true
I0 = 1
days = np.arange(0, 60)
I_clean = solve_sir(beta_true, gamma_true, I0, days, N)
I_obs = np.random.poisson(I_clean)

window_lengths = [20, 30, 40, 60]

# Existing (Jeffreys) results:
results_adaptive = batch_adaptive_for_windows_R0gamma(window_lengths, I_obs, days, I0, N)
batches = batch_adaptive_mcmc_all_chains_R0gamma(I_obs, days[:60], I0, N, n_batches=3, iter_per_batch=4000)

# STRONG prior results:
results_strong = batch_adaptive_for_windows_strong(window_lengths, I_obs, days, I0, N)
batches_strong = batch_adaptive_mcmc_all_chains_strong_gamma(I_obs, days[:60], I0, N,
                                                             n_batches=3, iter_per_batch=4000)

# Add R0 to both (so we can compare in (R0, gamma))
for w in window_lengths:
    gJ = np.maximum(results_adaptive[w]["gamma"], 1e-12)
    results_adaptive[w]["R0"] = results_adaptive[w]["beta"] / gJ

    gS = np.maximum(results_strong[w]["gamma"], 1e-12)
    results_strong[w]["R0"] = results_strong[w]["beta"] / gS






# Plot observed epidemic curve 
plt.figure(figsize=(10, 4))
plt.plot(days, I_obs, 'o-', label="Observed infected (Poisson)", alpha=0.7)
plt.plot(days, I_clean, '--', label="True underlying epidemic", color='black')
plt.title("Observed epidemic")
plt.xlabel("Time (days)")
plt.ylabel("Infected count")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()

R0_true = beta_true / gamma_true


# Joint posterior distribution plot
fig, axes = plt.subplots(2, len(window_lengths), figsize=(4 * len(window_lengths), 7), sharex=False, sharey=False)

for i, w in enumerate(window_lengths):
    # Jeffreys
    sns.kdeplot(
        x=results_adaptive[w]["R0"],
        y=results_adaptive[w]["gamma"],
        fill=True, cmap="Blues", ax=axes[0, i]
    )
    axes[0, i].axvline(R0_true, color='red', linestyle='--')
    axes[0, i].axhline(gamma_true, color='red', linestyle='--')
    axes[0, i].set_title(f"Jeffreys — {w} days")
    axes[0, i].set_xlabel("R₀"); axes[0, i].set_ylabel("γ")

    # Strong prior
    sns.kdeplot(
        x=results_strong[w]["R0"],
        y=results_strong[w]["gamma"],
        fill=True, cmap="Greens", ax=axes[1, i]
    )
    axes[1, i].axvline(R0_true, color='red', linestyle='--')
    axes[1, i].axhline(gamma_true, color='red', linestyle='--')
    axes[1, i].set_title(f"Log-normal — {w} days")
    axes[1, i].set_xlabel("R₀"); axes[1, i].set_ylabel("γ")

plt.tight_layout()
plt.show()

# Posterior density for R0 with both priors
fig, axes = plt.subplots(1, len(window_lengths), figsize=(4 * len(window_lengths), 3), sharey=True)
for i, w in enumerate(window_lengths):
    sns.kdeplot(results_adaptive[w]["R0"], fill=True, alpha=0.5, label="Jeffreys", ax=axes[i])
    sns.kdeplot(results_strong[w]["R0"], fill=True, alpha=0.5, label="Log-normal", ax=axes[i])
    axes[i].axvline(R0_true, color='red', linestyle='--', label='True R₀' if i==0 else None)
    axes[i].set_title(f"R₀ posterior — {w} days")
    axes[i].set_xlabel("R₀")
if len(window_lengths) > 0:
    axes[0].legend()
plt.tight_layout()
plt.show()

# Posterior density for R0 with the log-normal prior
fig, axes = plt.subplots(1, len(window_lengths), figsize=(4 * len(window_lengths), 3), sharey=True)
for i, w in enumerate(window_lengths):
    sns.kdeplot(results_strong[w]["R0"], fill=True, ax=axes[i], color="green", alpha=0.6)
    axes[i].axvline(R0_true, color='red', linestyle='--')
    axes[i].set_title(f"Strong γ prior — R₀ posterior (window={w})")
    axes[i].set_xlabel("R₀")
plt.tight_layout()
plt.show()

# Trace Plot Strong prior
fig, axes = plt.subplots(len(window_lengths), 2, figsize=(12, 2.5 * len(window_lengths)))
for i, w in enumerate(window_lengths):
    axes[i, 0].plot(results_strong[w]["R0"], color='Blue')
    axes[i, 0].set_title(f"Log-normal prior — Trace $R_0$ (window={w})")
    axes[i, 1].plot(results_strong[w]["gamma"], color='Blue')
    axes[i, 1].set_title(f"Log-normal prior — Trace γ (window={w})")
plt.xlabel("Iteration")
plt.tight_layout()
plt.show()


# Batch trace plot for the log-normal prior
fig, axes = plt.subplots(len(batches_strong), 2, figsize=(12, 2.5 * len(batches_strong)))
for i, chain in enumerate(batches_strong):
    axes[i, 0].plot(chain[:, 0])
    axes[i, 0].set_ylabel("β")
    axes[i, 0].set_title(f"Log-normal prior — Trace β (Batch {i+1})")
    axes[i, 0].set_xlabel("Iteration")

    axes[i, 1].plot(chain[:, 1])
    axes[i, 1].set_ylabel("γ")
    axes[i, 1].set_title(f"Log-normal prior — Trace γ (Batch {i+1})")
    axes[i, 1].set_xlabel("Iteration")
plt.tight_layout()
plt.show()

#Trace Plot for Jeffreys Prior
fig, axes = plt.subplots(len(window_lengths), 2, figsize=(12, 2.5 * len(window_lengths)))
for i, w in enumerate(window_lengths):
    axes[i, 0].plot(results_adaptive[w]["beta"], color='Blue')
    axes[i, 0].set_title(f"Jeffreys' prior — Trace β (window={w})")
    axes[i, 0].set_xlabel("Iteration")
    axes[i, 1].plot(results_adaptive[w]["gamma"], color='Blue')
    axes[i, 1].set_title(f"Jeffreys' prior — Trace γ (window={w})")
    axes[i, 1].set_xlabel("Iteration")

plt.tight_layout()
plt.show()

# Batch trace for Jeffreys prior
fig, axes = plt.subplots(len(batches), 2, figsize=(12, 2.5 * len(batches)))
for i, chain in enumerate(batches):
    axes[i, 0].plot(chain[:, 0])
    axes[i, 0].set_ylabel("β")
    axes[i, 0].set_title(f"Jeffreys' prior — Trace β (Batch {i+1})")
    axes[i, 0].set_xlabel("Iteration")

    axes[i, 1].plot(chain[:, 1])
    axes[i, 1].set_ylabel("γ")
    axes[i, 1].set_title(f"Jeffreys' prior — Trace γ (Batch {i+1})")
    axes[i, 1].set_xlabel("Iteration")
plt.tight_layout()
plt.show()
