import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import seaborn as sns

# -----------------------------
# SIR model + solver + log-post
# -----------------------------
def sir_model(t, y, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def solve_sir(beta, gamma, I0, t_eval, N):
    R0 = 0
    S0 = N - I0 - R0
    sol = solve_ivp(
        sir_model, [t_eval[0], t_eval[-1]], [S0, I0, R0],
        args=(beta, gamma, N), t_eval=t_eval, method='RK45'
    )
    return sol.y[1]  # prevalence I(t)

def poisson_log_likelihood(I_obs, I_model):
    I_model = np.maximum(I_model, 1e-6)  # guard
    return np.sum(I_obs * np.log(I_model) - I_model) / 1000  # tempered (×1/1000)

def log_prior(beta, gamma):
    if gamma <= 0 or beta <= 0:
        return -np.inf
    return 0.0  # flat over beta>0, gamma>0

def log_posterior(beta, gamma, I_obs, t_eval, I0, N):
    if beta <= 0 or gamma <= 0:
        return -np.inf
    I_model = solve_sir(beta, gamma, I0, t_eval, N)
    return poisson_log_likelihood(I_obs, I_model) + log_prior(beta, gamma)

#--------
# RWMH
# -------
def run_rwmh(I_obs, t_eval, I0, N, n_iter=10000,
             proposal_sd=0.05,          # used if sd_beta/sd_gamma are None
             sd_beta=None, sd_gamma=None,
             beta_init=0.5, gamma_init=0.1, rng=None):
    """
    Random-walk MH with Normal proposals and reflection at 0.
    If sd_beta/sd_gamma are provided, uses a diagonal proposal with those SDs.
    Otherwise uses a single scalar proposal_sd for both parameters.
    """
    rng = np.random.default_rng() if rng is None else rng
    beta, gamma = beta_init, gamma_init
    chain_beta, chain_gamma = [], []
    accept_count = 0

    # pick proposal SDs
    sdb = proposal_sd if sd_beta  is None else sd_beta
    sdg = proposal_sd if sd_gamma is None else sd_gamma

    log_post_curr = log_posterior(beta, gamma, I_obs, t_eval, I0, N)

    for _ in range(n_iter):
        # symmetric Normal RW + reflect at 0 via abs() to keep positivity
        beta_prop  = abs(rng.normal(beta,  sdb))
        gamma_prop = abs(rng.normal(gamma, sdg))

        log_post_prop = log_posterior(beta_prop, gamma_prop, I_obs, t_eval, I0, N)
        accept_ratio  = np.exp(log_post_prop - log_post_curr)

        if rng.random() < accept_ratio:
            beta, gamma = beta_prop, gamma_prop
            log_post_curr = log_post_prop
            accept_count += 1

        chain_beta.append(beta)
        chain_gamma.append(gamma)

    acceptance_rate = accept_count / n_iter
    return np.asarray(chain_beta), np.asarray(chain_gamma), acceptance_rate

# ---------- (R0, gamma) posterior + RWMH ----------
def log_posterior_r0_gamma(R0, gamma, I_obs, t_eval, I0, N, include_jacobian=True):
    """
    Uses beta = R0 * gamma. If include_jacobian=True, adds log(gamma), which
    makes this equivalent to a flat prior in (beta, gamma).
    If you want a flat prior in (R0, gamma) instead, set include_jacobian=False.
    """
    if R0 <= 0 or gamma <= 0:
        return -np.inf
    beta = R0 * gamma
    lp = log_posterior(beta, gamma, I_obs, t_eval, I0, N)
    if lp == -np.inf:
        return lp
    if include_jacobian:
        lp += np.log(gamma)  # Jacobian of (R0, gamma) -> (beta, gamma)
    return lp

def run_rwmh_r0_gamma(I_obs, t_eval, I0, N, n_iter=10000,
                      proposal_sd=0.05, sd_r0=None, sd_gamma=None,
                      R0_init=3.0, gamma_init=0.1, rng=None, include_jacobian=True):
    """
    Random-walk MH in (R0, gamma) with Normal proposals and reflection to keep positivity.
    """
    rng = np.random.default_rng() if rng is None else rng
    R0, gamma = R0_init, gamma_init
    chain_R0, chain_gamma = [], []
    accept_count = 0

    sdr = proposal_sd if sd_r0  is None else sd_r0
    sdg = proposal_sd if sd_gamma is None else sd_gamma

    log_post_curr = log_posterior_r0_gamma(R0, gamma, I_obs, t_eval, I0, N, include_jacobian)

    for _ in range(n_iter):
        R0_prop    = abs(rng.normal(R0,    sdr))
        gamma_prop = abs(rng.normal(gamma, sdg))

        log_post_prop = log_posterior_r0_gamma(R0_prop, gamma_prop, I_obs, t_eval, I0, N, include_jacobian)
        accept_ratio  = np.exp(log_post_prop - log_post_curr)

        if rng.random() < accept_ratio:
            R0, gamma = R0_prop, gamma_prop
            log_post_curr = log_post_prop
            accept_count += 1

        chain_R0.append(R0)
        chain_gamma.append(gamma)

    acceptance_rate = accept_count / n_iter
    return np.asarray(chain_R0), np.asarray(chain_gamma), acceptance_rate


# -----------------------------
# Simulate data (prevalence)
# -----------------------------
N = 1_000_000
beta_true = 0.6
gamma_true = 0.1
I0 = 1
full_days = np.arange(0, 60)

I_clean = solve_sir(beta_true, gamma_true, I0, full_days, N)
rng = np.random.default_rng(123)
I_obs = rng.poisson(I_clean)


# Per-window proposal SDs (manual)
sd_by_window = {20: 0.04, 30: 0.006, 40: 0.003, 60: 0.002}

# -----------------------------
# Fit per window (RWMH)
# -----------------------------
windows = [20, 30, 40, 60]
results = {}
burn = 1000

for w in windows:
    t_subset = full_days[:w]
    y_subset = I_obs[:w]

    # --- choose proposal SDs for this window (shared defaults) ---
    sd = sd_by_window.get(w, 0.02)

    # ===== (β, γ) sampler =====
    beta_chain, gamma_chain, acc_rate = run_rwmh(
        y_subset, t_subset, I0, N,
        n_iter=10000, proposal_sd=sd,
        beta_init=0.5, gamma_init=0.1, rng=rng
    )

    # ===== (R0, γ) sampler =====
    r0_chain, g_rg_chain, acc_rate_rg = run_rwmh_r0_gamma(
        y_subset, t_subset, I0, N,
        n_iter=10000, sd_r0=sd, sd_gamma=sd,
        R0_init=5, gamma_init=0.1, rng=rng, include_jacobian=True
    )

    # --- store + print ---
    results[w] = {
        "beta":  beta_chain[burn:],
        "gamma": gamma_chain[burn:],
        "accept_rate": acc_rate,
        "proposal_sd": sd,
        "R0": r0_chain[burn:],
        "gamma_rg": g_rg_chain[burn:],
        "accept_rate_rg": acc_rate_rg,
        "proposal_sd_r0": sd,
        "proposal_sd_g_rg": sd
    }
    print(f"[window {w}] (β,γ): sd={sd:.4f}  acceptance={acc_rate:.1%}")
    print(f"[window {w}] (R0,γ): sd_R0={sd:.4f} sd_γ={sd:.4f}  acceptance={acc_rate_rg:.1%}")

# -----------------------------
# Plots 
# -----------------------------
sns.set_style("white")
sns.set_context("notebook", font_scale=0.95)

# Trace plots (beta gamma) per window
fig, axes = plt.subplots(len(windows), 2, figsize=(10, 3 * len(windows)))
for i, w in enumerate(windows):
    b = results[w]["beta"]; g = results[w]["gamma"]; acc = results[w]["accept_rate"]
    axes[i, 0].plot(b, lw=0.9)
    axes[i, 0].set_title(f"Trace: β (window={w}d, acc={acc:.1%}, sd={results[w]['proposal_sd']:.3f})")
    axes[i, 0].set_xlabel("Iteration"); axes[i, 0].set_ylabel("β")
    axes[i, 1].plot(g, lw=0.9)
    axes[i, 1].set_title(f"Trace: γ (window={w}d)")
    axes[i, 1].set_xlabel("Iteration"); axes[i, 1].set_ylabel("γ")
plt.tight_layout(); plt.show()

# Joint posterior KDEs per window
fig, axes = plt.subplots(1, len(windows), figsize=(4 * len(windows), 4), sharex=False, sharey=False)
if len(windows) == 1: axes = [axes]
for i, w in enumerate(windows):
    b = results[w]["beta"]; g = results[w]["gamma"]
    sns.kdeplot(x=b, y=g, fill=True, levels=30, thresh=0.05, cmap="Blues", ax=axes[i])
    axes[i].axvline(beta_true,  color='red', ls='--', lw=1)
    axes[i].axhline(gamma_true, color='red', ls='--', lw=1)
    axes[i].set_title(f"Joint posterior (window={w}d)")
    axes[i].set_xlabel("β"); 
    if i == 0: axes[i].set_ylabel("γ")
plt.tight_layout(); plt.show()

# Marginal posteriors (β, γ) per window
fig, axes = plt.subplots(len(windows), 2, figsize=(10, 3 * len(windows)))
for i, w in enumerate(windows):
    b = results[w]["beta"]; g = results[w]["gamma"]
    sns.kdeplot(b, fill=True, ax=axes[i, 0])
    axes[i, 0].axvline(beta_true, color='red', ls='--', lw=1)
    axes[i, 0].set_title(f"β posterior (window={w}d)"); axes[i, 0].set_xlabel("β")
    sns.kdeplot(g, fill=True, ax=axes[i, 1])
    axes[i, 1].axvline(gamma_true, color='red', ls='--', lw=1)
    axes[i, 1].set_title(f"γ posterior (window={w}d)"); axes[i, 1].set_xlabel("γ")
plt.tight_layout(); plt.show()


# Trace plots (R0, γ) per window
fig, axes = plt.subplots(len(windows), 2, figsize=(10, 3 * len(windows)))
for i, w in enumerate(windows):
    r0 = results[w]["R0"]; g2 = results[w]["gamma_rg"]; acc = results[w]["accept_rate_rg"]
    axes[i, 0].plot(r0, lw=0.9)
    axes[i, 0].set_title(f"Trace: R₀ (window={w}d, acc={acc:.1%}, sd_R0={results[w]['proposal_sd_r0']:.3f})")
    axes[i, 0].set_xlabel("Iteration"); axes[i, 0].set_ylabel("R₀")
    axes[i, 1].plot(g2, lw=0.9)
    axes[i, 1].set_title(f"Trace: γ (R₀-param, window={w}d, sd_γ={results[w]['proposal_sd_g_rg']:.3f})")
    axes[i, 1].set_xlabel("Iteration"); axes[i, 1].set_ylabel("γ")
plt.tight_layout(); plt.show()

# Joint posterior KDEs for (R0, γ) per window
fig, axes = plt.subplots(1, len(windows), figsize=(4 * len(windows), 4), sharex=False, sharey=False)
if len(windows) == 1: axes = [axes]
for i, w in enumerate(windows):
    r0 = results[w]["R0"]; g2 = results[w]["gamma_rg"]
    sns.kdeplot(x=r0, y=g2, fill=True, levels=30, thresh=0.05, cmap="Greens", ax=axes[i])
    axes[i].axvline(beta_true/gamma_true,  color='red', ls='--', lw=1)
    axes[i].axhline(gamma_true,            color='red', ls='--', lw=1)
    axes[i].set_title(f"(R₀, γ) joint posterior (window={w}d)")
    axes[i].set_xlabel("R₀")
    if i == 0: axes[i].set_ylabel("γ")
plt.tight_layout(); plt.show()

# Marginal posteriors (R0, γ) per window for the (R0, γ) run
fig, axes = plt.subplots(len(windows), 2, figsize=(10, 3 * len(windows)))
for i, w in enumerate(windows):
    r0 = results[w]["R0"]; g2 = results[w]["gamma_rg"]
    sns.kdeplot(r0, fill=True, ax=axes[i, 0])
    axes[i, 0].axvline(beta_true/gamma_true, color='red', ls='--', lw=1)
    axes[i, 0].set_title(f"R₀ posterior (window={w}d)"); axes[i, 0].set_xlabel("R₀")
    sns.kdeplot(g2, fill=True, ax=axes[i, 1])
    axes[i, 1].axvline(gamma_true, color='red', ls='--', lw=1)
    axes[i, 1].set_title(f"γ posterior (R₀-param, window={w}d)"); axes[i, 1].set_xlabel("γ")
plt.tight_layout(); plt.show()

