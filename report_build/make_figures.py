"""Draw the report figures from the saved results (results/*.csv) and the notebook's own functions."""
import os, sys, json, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.special import expit
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import simulation_core as sc

plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5, "axes.titleweight": "bold"})
FIG = f"{ROOT}/figures"; os.makedirs(FIG, exist_ok=True)
MECHS = ["MCAR", "MAR", "MNAR"]
METHODS = ["Complete Case", "Mean Imputation", "Hot-Deck", "Regression Imputation", "Stochastic Regression", "MICE (normal)", "MICE (PMM)", "EM"]
ROWS = ["Full data"] + METHODS
GREY, ORANGE, BLUE = "#8c8c8c", "#e08a1e", "#2a7fb8"
FAMILY = {"Full data": "#cfcfcf", "Complete Case": GREY, "Mean Imputation": ORANGE, "Hot-Deck": ORANGE, "Regression Imputation": ORANGE,
          "Stochastic Regression": ORANGE, "MICE (normal)": BLUE, "MICE (PMM)": BLUE, "EM": BLUE}
KINDS = ["Classical", "Bootstrap standard", "Bootstrap percentile", "Rubin"]
KCOL = {"Classical": "#e08a1e", "Bootstrap standard": "#9ec3e6", "Bootstrap percentile": "#1f3a5f", "Rubin": "#2e8b57"}

mc = pd.read_csv(f"{ROOT}/results/monte_carlo_results.csv")
cov = pd.read_csv(f"{ROOT}/results/summary_coverage.csv").set_index(["Mechanism", "Method"])
hon = pd.read_csv(f"{ROOT}/results/summary_se_honesty.csv").set_index(["Mechanism", "Method"])
N_REPS = mc["Rep"].nunique(); MC_ERR = 100 * 1.96 * np.sqrt(0.95 * 0.05 / N_REPS)
full = sc.make_data(np.random.default_rng(sc.SEED))

def intervals(kind):
    d = mc[["Mechanism", "Method", "Rep", "Beta1"]].copy()
    if kind == "Classical": d["Low"], d["High"] = sc.normal_interval(mc.Beta1, mc.SE_classical)
    elif kind == "Bootstrap standard": d["Low"], d["High"] = sc.normal_interval(mc.Beta1, mc.SE_boot)
    elif kind == "Bootstrap percentile": d["Low"], d["High"] = mc.Pct_low, mc.Pct_high
    else: d["Low"], d["High"] = sc.normal_interval(mc.Beta1, mc.SE_rubin)
    d["Covered"] = (d.Low <= 2) & (2 <= d.High); return d.dropna(subset=["Low"])
IV = {k: intervals(k) for k in KINDS}
for k in KINDS:   # consistency check with the notebook's own summary
    chk = 100 * IV[k].groupby(["Mechanism", "Method"])["Covered"].mean()
    assert np.allclose(chk.sort_index(), cov[k].dropna().sort_index(), atol=0.01), k

# ---------- fig1: data and deletion rules
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1))
ax = axes[0]; ax.scatter(full.X1, full.Y, s=9, color=BLUE, alpha=0.8)
sl, ic = np.polyfit(full.X1, full.Y, 1); xs = np.array([full.X1.min(), full.X1.max()]); ax.plot(xs, ic + sl * xs, color="black", lw=1.5)
ax.set_xlabel("X1"); ax.set_ylabel("Y"); ax.set_title("(a) The complete data: Y rises with X1")
z = np.linspace(-3, 3, 200); ax = axes[1]
ax.plot(z, np.full_like(z, 0.20), color=GREY, lw=2.5, label="MCAR: always 20%"); ax.plot(z, expit(-2.23 + 2 * z), color="#c0392b", lw=2.5, label="MAR and MNAR: S-shaped curve")
ax.set_xlabel("the 'driver' (standardised)"); ax.set_ylabel("chance that the value is deleted"); ax.set_title("(b) How likely a value is to be deleted")
ax.legend(loc="upper left", frameon=False); ax.text(0.98, 0.30, "MAR: driver = Z\nMNAR: driver = the value\nthat gets deleted", transform=ax.transAxes, ha="right", fontsize=7.5, color="#444")
plt.tight_layout(); plt.savefig(f"{FIG}/fig1_design.png", dpi=220); plt.close()

# ---------- fig2: what each method did to the data (MAR)
rng = np.random.default_rng(sc.SEED); data = sc.make_missing(full, "MAR", np.random.default_rng(sc.SEED))
versions = {"Full data (truth)": full, "Complete Case": sc.drop_incomplete(data, rng), "Mean Imputation": sc.fill_mean(data, rng), "Hot-Deck": sc.fill_hot_deck(data, rng),
            "Regression Imputation": sc.fill_regression(data, rng), "Stochastic Regression": sc.fill_stochastic(data, rng),
            "MICE (normal), 1st imputation": sc.chained_impute(data, "normal", rng), "MICE (PMM), 1st imputation": sc.chained_impute(data, "pmm", rng)}
had_gap = data[["X1", "Y"]].isna().any(axis=1)
fig, axes = plt.subplots(2, 4, figsize=(7.8, 4.1), sharex=True, sharey=True); slopes = {}
for ax, (name, d) in zip(axes.ravel(), versions.items()):
    flag = had_gap.loc[d.index].to_numpy()
    ax.scatter(d.X1[~flag], d.Y[~flag], s=5, color=BLUE); ax.scatter(d.X1[flag], d.Y[flag], s=7, color=ORANGE)
    sl, ic = np.polyfit(d.X1, d.Y, 1); slopes[name] = round(sl, 2); xs = np.array([d.X1.min(), d.X1.max()])
    ax.plot(xs, ic + sl * xs, color="black", lw=1.2); ax.set_title(f"{name}\nslope = {sl:.2f}", fontsize=7); ax.tick_params(labelsize=7)
for ax in axes[1]: ax.set_xlabel("X1", fontsize=7.5)
for ax in axes[:, 0]: ax.set_ylabel("Y", fontsize=7.5)
plt.tight_layout(); plt.savefig(f"{FIG}/fig2_what_methods_do_MAR.png", dpi=220); plt.close()
json.dump(slopes, open(f"{HERE}/_scatter_slopes.json", "w"))

# ---------- fig3: one bootstrap (stochastic regression, MAR) - same random draws as notebook Step 6c
nbk = json.load(open(f"{ROOT}/notebook/missing_data_simulation.ipynb"))
est = {}
for name, f in [("Complete Case", sc.complete_case), ("Mean Imputation", sc.mean_imputation), ("Hot-Deck", sc.hot_deck), ("Regression Imputation", sc.regression_imputation),
                ("Stochastic Regression", sc.stochastic_regression), ("EM", sc.em)]:
    est[name] = f(data, np.random.default_rng(sc.SEED))     # estimates are not random for these, except stochastic (see below)
rng = np.random.default_rng(sc.SEED); demo = {}
funcs = {"Complete Case": sc.complete_case, "Mean Imputation": sc.mean_imputation, "Hot-Deck": sc.hot_deck, "Regression Imputation": sc.regression_imputation,
         "Stochastic Regression": sc.stochastic_regression, "EM": sc.em}
for name in METHODS:
    if name in funcs: demo[name] = sc.bootstrap_beta1(funcs[name], data, rng, sc.B_BOOT)
txt6c = next("".join(o.get("data", {}).get("text/plain", "")) for c in nbk["cells"] if c["cell_type"] == "code" and "".join(c["source"]).startswith("# ===== Step 6c") for o in c["outputs"] if "data" in o)
m = re.search(r"Stochastic Regression\s+([\d.]+)\s+\(([\d.]+), ([\d.]+)\)\s+\(([\d.]+), ([\d.]+)\)", txt6c)
p_nb = re.search(r"Stochastic Regression\s+\(([\d.]+), ([\d.]+)\)", txt6c.split("Bootstrap percentile")[-1])
b = demo["Stochastic Regression"]; p_lo, p_hi = sc.percentile_interval(b)
assert p_nb and abs(p_lo - float(p_nb.group(1))) < 6e-4 and abs(p_hi - float(p_nb.group(2))) < 6e-4, ("bootstrap demo does not match notebook", p_lo, p_hi, p_nb and p_nb.groups())
beta_s = float(m.group(1)); c_lo, c_hi = float(m.group(2)), float(m.group(3))
fig, ax = plt.subplots(figsize=(7.2, 3.2)); ax.hist(b, bins=30, color="#7fa8d1", edgecolor="white")
ax.axvline(beta_s, color="black", lw=1.5, label=f"estimate = {beta_s:.3f}")
ax.axvspan(c_lo, c_hi, color=ORANGE, alpha=0.18, label=f"classical interval ({c_lo:.3f}, {c_hi:.3f})")
ax.axvline(p_lo, color="#1f3a5f", ls="--", lw=1.5, label=f"bootstrap percentile interval ({p_lo:.3f}, {p_hi:.3f})"); ax.axvline(p_hi, color="#1f3a5f", ls="--", lw=1.5)
ax.axvline(2, color="#c0392b", ls=":", lw=1.8, label="true value 2")
ax.set_xlabel("β1 from one bootstrap resample"); ax.set_ylabel(f"number of resamples (of {sc.B_BOOT})"); ax.legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.savefig(f"{FIG}/fig3_one_bootstrap.png", dpi=220); plt.close()
json.dump({"beta": beta_s, "classical": [c_lo, c_hi], "pct": [p_lo, p_hi], "boot_se": float(b.std(ddof=1))}, open(f"{HERE}/_boot_demo.json", "w"))

# ---------- fig4-6: per mechanism, three panels
def mech_fig(mech, n):
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 4.6), sharey=True); y = np.arange(len(ROWS))
    groups = [mc[(mc.Mechanism == mech) & (mc.Method == m)].Beta1.to_numpy() for m in ROWS]
    bp = axes[0].boxplot(groups, positions=y, orientation="horizontal", patch_artist=True, showfliers=False, widths=0.6)
    for p_, m_ in zip(bp["boxes"], ROWS): p_.set_facecolor(FAMILY[m_])
    for med in bp["medians"]: med.set_color("black")
    axes[0].axvline(2, color="#c0392b", ls="--", lw=1.3); axes[0].set_yticks(y); axes[0].set_yticklabels(ROWS); axes[0].invert_yaxis()
    axes[0].set_title("(a) Spread of the β1\nestimates (red = truth)"); axes[0].set_xlabel("β1 estimate")
    h = 0.2
    for k, kind in enumerate(KINDS):
        v = cov.loc[mech].loc[ROWS, kind]; ok = v.notna().to_numpy()
        axes[1].barh(y[ok] + (k - 1.5) * h, v[ok], h, color=KCOL[kind], label=kind)
    axes[1].axvline(95, color="black", ls="--", lw=1); axes[1].axvspan(95 - MC_ERR, 95 + MC_ERR, color="grey", alpha=0.25); axes[1].set_xlim(0, 100)
    axes[1].set_title("(b) Coverage of each\n95% interval (target 95%)"); axes[1].set_xlabel("coverage (%)")
    h3 = 0.27; SEC = {"Classical SE / SD": KCOL["Classical"], "Bootstrap SE / SD": KCOL["Bootstrap percentile"], "Rubin SE / SD": KCOL["Rubin"]}
    for k, col in enumerate(SEC):
        v = hon.loc[mech].loc[ROWS, col]; ok = v.notna().to_numpy()
        axes[2].barh(y[ok] + (k - 1) * h3, v[ok], h3, color=SEC[col])
    axes[2].axvline(1, color="black", ls="--", lw=1); axes[2].set_xlim(0, 1.35)
    axes[2].set_title("(c) Reported SE ÷ true\nspread (1 = honest)"); axes[2].set_xlabel("ratio")
    handles = [Patch(color=KCOL[k], label=k) for k in KINDS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.5, title="interval / standard error type", title_fontsize=7.5)
    plt.tight_layout(rect=[0, 0.09, 1, 1]); plt.savefig(f"{FIG}/fig{n}_simulation_{mech}.png", dpi=220); plt.close()
for mech, n in zip(MECHS, (4, 5, 6)): mech_fig(mech, n)

# ---------- fig7: individual intervals, classical vs bootstrap percentile (MAR)
fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.6), sharex=True)
for r, m_ in enumerate(["Complete Case", "Stochastic Regression"]):
    for c_, kind in enumerate(["Classical", "Bootstrap percentile"]):
        ax = axes[r, c_]; t = IV[kind][(IV[kind].Mechanism == "MAR") & (IV[kind].Method == m_) & (IV[kind].Rep < 40)]
        ax.hlines(t.Rep, t.Low, t.High, colors=np.where(t.Covered, "#2e8b57", "#c0392b"), lw=1.3); ax.plot(t.Beta1, t.Rep, "k.", ms=2.5)
        ax.axvline(2, color="black", ls="--"); ax.set_yticks([])
        ax.set_title(f"{m_}: {kind}\ncoverage over all {N_REPS} runs: {cov.loc[('MAR', m_), kind]:.0f}%", fontsize=8)
for ax in axes[1]: ax.set_xlabel("β1 estimate and 95% interval")
plt.tight_layout(); plt.savefig(f"{FIG}/fig7_intervals_MAR.png", dpi=220); plt.close()

# ---------- fig8: coverage heat maps, without and with bootstrap
def grid(kind):
    g = pd.DataFrame(index=ROWS, columns=MECHS, dtype=float)
    for me in MECHS:
        for m_ in ROWS: g.loc[m_, me] = cov.loc[(me, m_), "Rubin" if m_.startswith("MICE") else kind]
    return g
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.9))
for ax, (kind, title) in zip(axes, [("Classical", "Without bootstrap\n(classical; Rubin for MICE)"), ("Bootstrap percentile", "With bootstrap\n(percentile; Rubin for MICE)")]):
    g = grid(kind); im = ax.imshow(g.to_numpy(dtype=float), cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(MECHS); ax.set_yticks(range(len(g))); ax.set_yticklabels(g.index if ax is axes[0] else [])
    for i in range(g.shape[0]):
        for j in range(3):
            v = g.iloc[i, j]; dark = (not np.isnan(v)) and (v >= 85 or v <= 15)
            ax.text(j, i, "–" if np.isnan(v) else f"{v:.0f}%", ha="center", va="center", fontsize=7.5, color="white" if dark else "black")
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(title)
plt.tight_layout(); plt.savefig(f"{FIG}/fig8_coverage_before_after.png", dpi=220); plt.close()
print("figures:", sorted(os.listdir(FIG)))
