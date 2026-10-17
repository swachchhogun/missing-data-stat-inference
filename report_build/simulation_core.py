"""The same simulation functions as in the notebook, written as a plain .py file.
Used by make_figures.py and build_pdf.py to redraw the report figures."""
import numpy as np, pandas as pd
from scipy.special import expit
from sklearn.impute import KNNImputer

TRUE_BETA = np.array([5.0, 2.0, -1.5, 1.0])
N = 200
COLS = ["X1", "X2", "X3", "Z", "Y"]
M_IMPUTATIONS = 20
EM_BOOT = 50

def make_data(rng, n=N):
    X = rng.normal(size=(n, 3))
    Y = TRUE_BETA[0] + X @ TRUE_BETA[1:] + rng.normal(size=n)
    Z = Y + rng.normal(size=n)
    return pd.DataFrame(np.column_stack([X, Z, Y]), columns=COLS)

def standardize(s): return (s - s.mean()) / s.std()

def make_missing(full, mechanism, rng):
    data = full.copy()
    for col in ["Y", "X1"]:
        if mechanism == "MCAR":
            prob = 0.20
        else:
            source = full["Z"] if mechanism == "MAR" else full[col]
            prob = expit(-2.23 + 2.0 * standardize(source))
        data.loc[rng.random(len(full)) < prob, col] = np.nan
    return data

def ols(d):
    A = np.column_stack([np.ones(len(d)), d[["X1","X2","X3"]].to_numpy()])
    y = d["Y"].to_numpy()
    AtA_inv = np.linalg.inv(A.T @ A)
    beta = AtA_inv @ A.T @ y
    resid = y - A @ beta
    s2 = resid @ resid / (len(y) - A.shape[1])
    return beta, np.sqrt(s2 * np.diag(AtA_inv))

def hot_deck_fill(data):
    mean, sd = data.mean(), data.std()
    filled = KNNImputer(n_neighbors=1).fit_transform((data - mean) / sd)
    return pd.DataFrame(filled, columns=data.columns) * sd + mean

def chained_impute(data, kind, rng, cycles=5):
    X = data.to_numpy().copy()
    holes = np.isnan(X)
    X = np.where(holes, np.nanmean(X, axis=0), X)
    for _ in range(cycles):
        for j in np.where(holes.any(axis=0))[0]:
            obs, mis = ~holes[:, j], holes[:, j]
            A = np.column_stack([np.ones(len(X)), np.delete(X, j, axis=1)])
            beta, *_ = np.linalg.lstsq(A[obs], X[obs, j], rcond=None)
            resid = X[obs, j] - A[obs] @ beta
            if kind == "regression":
                X[mis, j] = A[mis] @ beta
            elif kind == "stochastic":
                X[mis, j] = A[mis] @ beta + rng.choice(resid, mis.sum())
            else:
                df_res = obs.sum() - A.shape[1]
                sigma2 = (resid @ resid / df_res) * df_res / rng.chisquare(df_res)
                beta_new = rng.multivariate_normal(beta, sigma2 * np.linalg.inv(A[obs].T @ A[obs]))
                if kind == "normal":
                    X[mis, j] = A[mis] @ beta_new + rng.normal(0, np.sqrt(sigma2), mis.sum())
                else:
                    pred_obs, pred_mis = A[obs] @ beta, A[mis] @ beta_new
                    nearest = np.argsort(np.abs(pred_mis[:, None] - pred_obs[None, :]), axis=1)[:, :5]
                    donor = nearest[np.arange(mis.sum()), rng.integers(0, 5, mis.sum())]
                    X[mis, j] = X[obs, j][donor]
    return pd.DataFrame(X, columns=data.columns)

def em_mvn(data, max_iter=200, tol=1e-6):
    X = data.to_numpy(); n, p = X.shape
    miss = np.isnan(X)
    patterns = np.unique(miss, axis=0)
    mu = np.nanmean(X, axis=0); S = np.diag(np.nanvar(X, axis=0))
    for _ in range(max_iter):
        Xhat = X.copy(); extra = np.zeros((p, p))
        for pat in patterns:
            if not pat.any(): continue
            rows = (miss == pat).all(axis=1)
            m, o = pat, ~pat
            B = S[np.ix_(m, o)] @ np.linalg.inv(S[np.ix_(o, o)])
            Xhat[np.ix_(rows, m)] = mu[m] + (X[np.ix_(rows, o)] - mu[o]) @ B.T
            extra[np.ix_(m, m)] += rows.sum() * (S[np.ix_(m, m)] - B @ S[np.ix_(o, m)])
        mu_new = Xhat.mean(axis=0)
        D = Xhat - mu_new
        S_new = (D.T @ D + extra) / n
        done = max(np.abs(mu_new - mu).max(), np.abs(S_new - S).max()) < tol
        mu, S = mu_new, S_new
        if done: break
    return mu, S

def moments_to_beta(mu, S):
    x, y = [0, 1, 2], 4
    b = np.linalg.solve(S[np.ix_(x, x)], S[x, y])
    return np.r_[mu[y] - b @ mu[x], b]

def rubin(d, kind, rng, m=M_IMPUTATIONS):
    fits = [ols(chained_impute(d, kind, rng)) for _ in range(m)]
    betas = np.array([f[0] for f in fits]); var = np.array([f[1]**2 for f in fits])
    total = var.mean(0) + (1 + 1/m) * betas.var(0, ddof=1)
    return betas.mean(0), np.sqrt(total)

def cc(d, rng):             return ols(d.dropna())
def mean_imp(d, rng):       return ols(d.fillna(d.mean()))
def hotdeck(d, rng):        return ols(hot_deck_fill(d))
def reg_imp(d, rng):        return ols(chained_impute(d, "regression", rng))
def stoch(d, rng):          return ols(chained_impute(d, "stochastic", rng))
def mice_normal(d, rng):    return rubin(d, "normal", rng)
def mice_pmm(d, rng):       return rubin(d, "pmm", rng)
def em(d, rng):             return moments_to_beta(*em_mvn(d)), None

METHODS = {"Complete Case": cc, "Mean Imputation": mean_imp, "Hot-Deck": hotdeck,
           "Regression Imputation": reg_imp, "Stochastic Regression": stoch,
           "MICE (normal)": mice_normal, "MICE (PMM)": mice_pmm, "EM": em}

def bootstrap(method, d, rng, B):
    est = []
    for _ in range(B):
        sample = d.iloc[rng.integers(0, len(d), len(d))].reset_index(drop=True)
        est.append(method(sample, rng)[0])
    return np.array(est)
