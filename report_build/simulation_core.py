"""Generated from the notebook by extract_core.py - do not edit by hand."""

# ===== Step 1a — Libraries =====
# If a package is missing, uncomment the next line:
# !pip install numpy pandas scipy scikit-learn statsmodels matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.special import expit
from sklearn.impute import KNNImputer


# ===== Step 1b — Settings you can change =====

SEED = 42                                      # seed for the one-dataset walkthrough (Sections 2-6)
N = 200                                        # observations per dataset
TRUE_BETA = np.array([5.0, 2.0, -1.5, 1.0])    # intercept, X1, X2, X3
M_IMPUTATIONS = 10                             # number of imputed datasets in MICE

N_REPS = 300                                   # Monte Carlo repetitions (new datasets from the true model)
B_BOOT = 200                                   # bootstrap resamples inside EVERY simulated dataset
MC_SEED = 2024                                 # seed for the Monte Carlo study
N_JOBS = -1                                    # CPU cores for the simulation: -1 = all cores, 1 = no parallel computing
USE_SAVED_RESULTS = True                       # reuse saved simulation results if they exist with these same settings
# Quick test: set N_REPS = 20 and B_BOOT = 50 (takes about a minute).
# The full study (300 x 200) takes roughly 40 minutes on one core, less with several cores.

MECHANISMS = ["MCAR", "MAR", "MNAR"]
METHOD_ORDER = ["Complete Case", "Mean Imputation", "Hot-Deck", "Regression Imputation",
                "Stochastic Regression", "MICE (normal)", "MICE (PMM)", "EM"]

# results are saved next to the notebook (or in the repository's results/ folder)

# ===== Step 2a — Function that creates one complete dataset =====
def make_data(rng, n=N):
    """One complete dataset: predictors X1-X3, outcome Y, auxiliary variable Z."""
    X = rng.normal(size=(n, 3))
    Y = TRUE_BETA[0] + X @ TRUE_BETA[1:] + rng.normal(size=n)
    Z = Y + rng.normal(size=n)                      # noisy copy of Y (not part of the regression)
    return pd.DataFrame(np.column_stack([X, Z, Y]), columns=["X1", "X2", "X3", "Z", "Y"])

# ===== Step 3a — The regression helper =====
def ols(d):
    """Least squares of Y on X1, X2, X3. Returns (coefficients, standard errors).
    The rows must have no missing values."""
    A = np.column_stack([np.ones(len(d)), d[["X1", "X2", "X3"]].to_numpy()])
    y = d["Y"].to_numpy()
    AtA_inv = np.linalg.inv(A.T @ A)
    beta = AtA_inv @ A.T @ y
    resid = y - A @ beta
    sigma2 = resid @ resid / (len(y) - A.shape[1])
    return beta, np.sqrt(sigma2 * np.diag(AtA_inv))

# ===== Step 4a — Helper, and the MCAR deletion rule =====
def standardize(s):
    """Rescale to mean 0 and standard deviation 1."""
    return (s - s.mean()) / s.std()


def delete_mcar(full, rng):
    """MCAR: every value of Y and X1 has the same 20% chance of being deleted."""
    data = full.copy()
    for col in ["Y", "X1"]:
        data.loc[rng.random(len(full)) < 0.20, col] = np.nan
    return data

# ===== Step 4b — The MAR deletion rule =====
def delete_mar(full, rng):
    """MAR: the chance of deletion depends on Z, which is always observed."""
    data = full.copy()
    prob = expit(-2.23 + 2.0 * standardize(full["Z"]))          # larger Z -> more likely to be deleted
    for col in ["Y", "X1"]:
        data.loc[rng.random(len(full)) < prob, col] = np.nan
    return data

# ===== Step 4c — The MNAR deletion rule =====
def delete_mnar(full, rng):
    """MNAR: the chance of deletion depends on the very value that gets deleted."""
    data = full.copy()
    for col in ["Y", "X1"]:
        prob = expit(-2.23 + 2.0 * standardize(full[col]))      # larger value -> more likely to be deleted
        data.loc[rng.random(len(full)) < prob, col] = np.nan
    return data

# ===== Step 4d — One function that applies any of the three rules =====
def make_missing(full, mechanism, rng):
    rules = {"MCAR": delete_mcar, "MAR": delete_mar, "MNAR": delete_mnar}
    return rules[mechanism](full, rng)

# ===== Step 5.2a — Complete case: the method =====
def drop_incomplete(data, rng):
    """Remove every row that has at least one missing value."""
    return data.dropna()


def complete_case(data, rng):
    """Complete-case analysis = drop the incomplete rows, then run the regression."""
    return ols(drop_incomplete(data, rng))

# ===== Step 5.3a — Mean imputation: the method =====
def fill_mean(data, rng):
    """Replace every gap by the mean of the observed values in that column."""
    return data.fillna(data.mean())                  # data.mean() skips the gaps


def mean_imputation(data, rng):
    return ols(fill_mean(data, rng))

# ===== Step 5.4a — Hot-deck: the method =====
def fill_hot_deck(data, rng):
    """Copy every missing value from the most similar row (one nearest neighbour)."""
    mean, sd = data.mean(), data.std()
    standardised = (data - mean) / sd                                      # put all variables on the same scale
    filled = KNNImputer(n_neighbors=1).fit_transform(standardised)         # 1 neighbour = 1 donor
    return pd.DataFrame(filled, columns=data.columns) * sd + mean          # back to the original scale


def hot_deck(data, rng):
    return ols(fill_hot_deck(data, rng))

# ===== Step 5.5a — Fit the regression for one incomplete column =====
def regress_column(X, holes, j):
    """Regress column j on all the other columns, using only the rows where column j is observed."""
    obs, mis = ~holes[:, j], holes[:, j]
    A = np.column_stack([np.ones(len(X)), np.delete(X, j, axis=1)])        # predictors = all other columns
    beta, *_ = np.linalg.lstsq(A[obs], X[obs, j], rcond=None)
    resid = X[obs, j] - A[obs] @ beta
    return A, beta, resid, obs, mis

# ===== Step 5.5b — MICE ingredient: draw new coefficients =====
def draw_new_coefficients(A, obs, beta, resid, rng):
    """Instead of treating the fitted coefficients as exact, draw new ones from their uncertainty."""
    df_res = obs.sum() - A.shape[1]
    sigma2 = resid @ resid / rng.chisquare(df_res)                                        # new error variance
    beta_new = rng.multivariate_normal(beta, sigma2 * np.linalg.inv(A[obs].T @ A[obs]))   # new coefficients
    return beta_new, sigma2

# ===== Step 5.5c — MICE ingredient: predictive mean matching =====
def pmm_donors(pred_obs, pred_mis, y_obs, rng, k=5):
    """For every gap, find the k observed rows whose predicted value is closest,
    pick one of them at random and copy its REAL observed value."""
    nearest = np.argsort(np.abs(pred_mis[:, None] - pred_obs[None, :]), axis=1)[:, :k]
    chosen = nearest[np.arange(len(pred_mis)), rng.integers(0, k, len(pred_mis))]
    return y_obs[chosen]

# ===== Step 5.5d — The engine itself =====
def chained_impute(data, kind, rng, cycles=5):
    """Fill the gaps one variable at a time, cycling through the incomplete variables.
    kind = "regression" | "stochastic" | "normal" | "pmm"."""
    X = data.to_numpy().copy()
    holes = np.isnan(X)
    X = np.where(holes, np.nanmean(X, axis=0), X)                      # start with simple mean imputation

    for _ in range(cycles):
        for j in np.where(holes.any(axis=0))[0]:                       # each incomplete variable in turn
            A, beta, resid, obs, mis = regress_column(X, holes, j)

            if kind == "regression":                                   # the predicted value
                X[mis, j] = A[mis] @ beta
            elif kind == "stochastic":                                 # predicted value + a random residual
                X[mis, j] = A[mis] @ beta + rng.choice(resid, mis.sum())
            else:                                                      # MICE: first draw new coefficients
                beta_new, sigma2 = draw_new_coefficients(A, obs, beta, resid, rng)
                if kind == "normal":                                   # a draw from the regression model
                    X[mis, j] = A[mis] @ beta_new + rng.normal(0, np.sqrt(sigma2), mis.sum())
                else:                                                  # "pmm": a real value from a similar row
                    X[mis, j] = pmm_donors(A[obs] @ beta, A[mis] @ beta_new, X[obs, j], rng)

    return pd.DataFrame(X, columns=data.columns)

# ===== Step 5.6a — Regression imputation: the method =====
def fill_regression(data, rng):
    return chained_impute(data, "regression", rng)


def regression_imputation(data, rng):
    return ols(fill_regression(data, rng))

# ===== Step 5.7a — Stochastic regression: the method =====
def fill_stochastic(data, rng):
    return chained_impute(data, "stochastic", rng)


def stochastic_regression(data, rng):
    return ols(fill_stochastic(data, rng))

# ===== Step 5.8a — MICE, step 1: create m completed datasets =====
def mice_impute(data, kind, rng):
    return [chained_impute(data, kind, rng) for _ in range(M_IMPUTATIONS)]

# ===== Step 5.8b — MICE, step 2: fit the regression on every completed dataset =====
def fit_each(completed):
    """Returns the coefficients and the sampling variances, one row per completed dataset."""
    fits = [ols(d) for d in completed]
    return np.array([f[0] for f in fits]), np.array([f[1] ** 2 for f in fits])

# ===== Step 5.8c — MICE, step 3: pool with Rubin's rules =====
def pool_rubin(betas, variances):
    m = len(betas)
    within = variances.mean(axis=0)                     # average sampling variance
    between = betas.var(axis=0, ddof=1)                 # disagreement between the m completed datasets
    total = within + (1 + 1 / m) * between
    return betas.mean(axis=0), np.sqrt(total), within, between

# ===== Step 5.8d — The two MICE methods as one-line functions =====
def mice_normal(data, rng):
    betas, variances = fit_each(mice_impute(data, "normal", rng))
    beta, se, _, _ = pool_rubin(betas, variances)
    return beta, se


def mice_pmm(data, rng):
    betas, variances = fit_each(mice_impute(data, "pmm", rng))
    beta, se, _, _ = pool_rubin(betas, variances)
    return beta, se


def peek_many(data, completed, truth, col="Y", n=5, shown=3):
    """True values of the first deleted entries of `col`, next to what the first few imputations filled in."""
    gone = data[col].isna()
    table = pd.DataFrame({"true value": truth.loc[gone, col]})
    for k in range(shown):
        table[f"imputation {k + 1}"] = completed[k].loc[gone, col]
    return table.head(n).round(2)

# ===== Step 5.9a — The EM algorithm =====
def em_mvn(data, max_iter=200, tol=1e-5):
    """EM for the mean vector (mu) and covariance matrix (S) of a multivariate normal with missing values.
    Rows with the same pattern of missingness are handled together."""
    X = data.to_numpy()
    n, p = X.shape
    miss = np.isnan(X)
    patterns = np.unique(miss, axis=0)                              # the distinct missingness patterns
    mu, S = np.nanmean(X, axis=0), np.diag(np.nanvar(X, axis=0))    # starting values

    for _ in range(max_iter):
        # E-step: replace each gap by its expected value given the observed values
        X_filled, extra = X.copy(), np.zeros((p, p))
        for pattern in patterns:
            if not pattern.any():
                continue                                            # complete rows need nothing
            rows, m, o = (miss == pattern).all(axis=1), pattern, ~pattern
            B = S[np.ix_(m, o)] @ np.linalg.inv(S[np.ix_(o, o)])    # regression of missing on observed
            X_filled[np.ix_(rows, m)] = mu[m] + (X[np.ix_(rows, o)] - mu[o]) @ B.T
            extra[np.ix_(m, m)] += rows.sum() * (S[np.ix_(m, m)] - B @ S[np.ix_(o, m)])
        # M-step: re-estimate mu and S from the completed data
        mu_new = X_filled.mean(axis=0)
        centred = X_filled - mu_new
        S_new = (centred.T @ centred + extra) / n
        converged = max(np.abs(mu_new - mu).max(), np.abs(S_new - S).max()) < tol
        mu, S = mu_new, S_new
        if converged:
            break
    return mu, S

# ===== Step 5.9b — From means and covariances to regression coefficients =====
def moments_to_beta(mu, S):
    """Regression coefficients implied by the means and covariances.
    Columns are X1, X2, X3, Z, Y, so the predictors are 0-2 and the outcome is column 4."""
    x, y = [0, 1, 2], 4
    slopes = np.linalg.solve(S[np.ix_(x, x)], S[x, y])
    return np.r_[mu[y] - slopes @ mu[x], slopes]


def em(data, rng):
    """EM as a method: returns the coefficients (and None, because EM has no simple standard error)."""
    return moments_to_beta(*em_mvn(data)), None

# ===== Step 6a — The bootstrap: resample the rows and redo the WHOLE method =====
def bootstrap_beta1(method, data, rng, B):
    """B bootstrap estimates of beta1. Each resample draws n rows with replacement
    and runs the complete method on it, filling-in of the gaps included."""
    n = len(data)
    estimates_b = np.empty(B)
    for b in range(B):
        resample = data.iloc[rng.integers(0, n, n)].reset_index(drop=True)
        estimates_b[b] = method(resample, rng)[0][1]
    return estimates_b

# ===== Step 6b — From an estimate and its SE (or bootstrap estimates) to an interval =====
def normal_interval(estimate, se):
    """Classical, bootstrap-standard and Rubin intervals all have this form."""
    return estimate - 1.96 * se, estimate + 1.96 * se


def percentile_interval(boot):
    """Bootstrap percentile interval: the 2.5% and 97.5% points of the bootstrap estimates."""
    return np.percentile(boot, 2.5), np.percentile(boot, 97.5)
