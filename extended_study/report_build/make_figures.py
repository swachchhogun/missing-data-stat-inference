import json, base64, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.special import expit
import os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from simulation_core import make_data, ols

plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.titleweight": "bold", "figure.dpi": 100})
MECHS = ["MCAR", "MAR", "MNAR"]
METHODS = ["Complete Case", "Mean Imputation", "Hot-Deck", "Regression Imputation",
           "Stochastic Regression", "MICE (normal)", "MICE (PMM)", "EM"]
GREY, ORANGE, BLUE = "#8c8c8c", "#e08a1e", "#2a7fb8"
FAMILY = {"Complete Case": GREY, "Mean Imputation": ORANGE, "Hot-Deck": ORANGE, "Regression Imputation": ORANGE,
          "Stochastic Regression": ORANGE, "MICE (normal)": BLUE, "MICE (PMM)": BLUE, "EM": BLUE, "Full data": "#cfcfcf"}
LEGEND = [Patch(color=GREY, label="Delete incomplete rows"), Patch(color=ORANGE, label="Single imputation"),
          Patch(color=BLUE, label="Multiple imputation / EM"), Patch(color="#cfcfcf", label="No missing data (benchmark)")]

s = pd.read_csv(f"{ROOT}/results/simulation_summary.csv"); a = pd.read_csv(f"{ROOT}/results/part_a_results.csv"); sim = pd.read_csv(f"{ROOT}/results/simulation_all_repetitions.csv")
full = make_data(np.random.default_rng(42)); beta_full = ols(full)[0]
assert abs(beta_full[1] - 2.0368) < 1e-3, beta_full

# ---------- Fig 1: the complete data and the deletion rules
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1))
ax = axes[0]
ax.scatter(full["X1"], full["Y"], s=9, color=BLUE, alpha=0.8)
xs = np.array([full.X1.min(), full.X1.max()]); sl, ic = np.polyfit(full.X1, full.Y, 1)
ax.plot(xs, ic + sl * xs, color="black", lw=1.5)
ax.set_xlabel("X1"); ax.set_ylabel("Y"); ax.set_title("(a) The complete data: Y rises with X1")
z = np.linspace(-3, 3, 200)
ax = axes[1]
ax.plot(z, np.full_like(z, 0.20), color=GREY, lw=2.5, label="MCAR: always 20%")
ax.plot(z, expit(-2.23 + 2 * z), color="#c0392b", lw=2.5, label="MAR and MNAR: S-shaped curve")
ax.set_xlabel("the 'driver' (standardised)"); ax.set_ylabel("chance that the value is deleted")
ax.set_title("(b) How likely a value is to be deleted"); ax.legend(loc="upper left", frameon=False)
ax.text(0.98, 0.30, "MAR: driver = Z\nMNAR: driver = the value\nthat gets deleted", transform=ax.transAxes, ha="right", fontsize=7.5, color="#444")
plt.tight_layout(); plt.savefig(f"{ROOT}/figures/fig1_design.png", dpi=200); plt.close()

# ---------- Fig 2: Part A forest plot (3 panels)
fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.5), sharex=True, sharey=True)
for ax, mech in zip(axes, MECHS):
    t = a[a.Mechanism == mech].set_index("Method").loc[METHODS]; y = np.arange(len(t))
    for yi, (m, r) in zip(y, t.iterrows()):
        ax.errorbar(r.Beta1, yi, xerr=[[r.Beta1 - r.CI_Low], [r.CI_High - r.Beta1]], fmt="o", color=FAMILY[m], capsize=4, lw=2)
    ax.axvline(2, color="#c0392b", ls="--", lw=1.3); ax.axvline(beta_full[1], color="black", ls=":", lw=1.3)
    ax.set_yticks(y); ax.set_yticklabels(METHODS); ax.set_title(mech); ax.set_xlabel("β1 estimate (95% interval)")
axes[0].invert_yaxis()
axes[0].plot([], [], color="#c0392b", ls="--", label="true β1 = 2"); axes[0].plot([], [], color="black", ls=":", label="full-data estimate")
axes[0].legend(loc="lower left", frameon=False, fontsize=6.5)
plt.tight_layout(); plt.savefig(f"{ROOT}/figures/fig2_partA_intervals.png", dpi=200); plt.close()

# ---------- Fig 3: what each method did to the data (MAR dataset), drawn at print size
from simulation_core import make_missing, hot_deck_fill, chained_impute
rng = np.random.default_rng(42)
data = make_missing(full, "MAR", np.random.default_rng(42))
versions = {"Full data (truth)": full, "Complete Case": data.dropna(), "Mean Imputation": data.fillna(data.mean()),
            "Hot-Deck": hot_deck_fill(data), "Regression Imputation": chained_impute(data, "regression", rng),
            "Stochastic Regression": chained_impute(data, "stochastic", rng), "MICE (normal), 1st imputation": chained_impute(data, "normal", rng),
            "MICE (PMM), 1st imputation": chained_impute(data, "pmm", rng)}
had_gap = data[["X1", "Y"]].isna().any(axis=1)
fig, axes = plt.subplots(2, 4, figsize=(7.8, 4.1), sharex=True, sharey=True)
slopes = {}
for ax, (name, d) in zip(axes.ravel(), versions.items()):
    flag = had_gap.loc[d.index].to_numpy()
    ax.scatter(d["X1"][~flag], d["Y"][~flag], s=5, color="#2a7fb8"); ax.scatter(d["X1"][flag], d["Y"][flag], s=7, color="#e08a1e")
    sl, ic = np.polyfit(d["X1"], d["Y"], 1); slopes[name] = round(sl, 2); xs = np.array([d["X1"].min(), d["X1"].max()])
    ax.plot(xs, ic + sl * xs, color="black", lw=1.2); ax.set_title(f"{name}\nslope = {sl:.2f}", fontsize=7)
    ax.tick_params(labelsize=7)
for ax in axes[1]: ax.set_xlabel("X1", fontsize=7.5)
for ax in axes[:, 0]: ax.set_ylabel("Y", fontsize=7.5)
plt.tight_layout(); plt.savefig(f"{ROOT}/figures/fig3_what_methods_do_MAR.png", dpi=220); plt.close()
print("scatter slopes:", slopes)

# ---------- Fig 4-6: simulation results, one figure per mechanism (3 panels)
def mech_figure(mech, n):
    t = s[s.Mechanism == mech].set_index("Method"); order = ["Full data"] + METHODS; y = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 4.2), sharey=True)
    # (a) distribution of the estimates
    groups = [sim[(sim.Mechanism == mech) & (sim.Method == m)].Beta1.to_numpy() for m in order]
    bp = axes[0].boxplot(groups, positions=y, vert=False, patch_artist=True, showfliers=False, widths=0.6)
    for patch, m in zip(bp["boxes"], order): patch.set_facecolor(FAMILY[m]); patch.set_alpha(0.9)
    for med in bp["medians"]: med.set_color("black")
    axes[0].axvline(2, color="#c0392b", ls="--", lw=1.4)
    axes[0].set_yticks(y); axes[0].set_yticklabels(order); axes[0].invert_yaxis()
    axes[0].set_title("(a) Spread of the β1\nestimates (red = truth)"); axes[0].set_xlabel("β1 estimate")
    # (b) coverage
    cov = t.loc[order, "Coverage (%)"]; mc = 1.96 * np.sqrt(0.95 * 0.05 / 300) * 100
    axes[1].barh(y, cov, color=[FAMILY[m] for m in order]); axes[1].axvline(95, color="black", ls="--", lw=1.2)
    axes[1].axvspan(95 - mc, 95 + mc, color="grey", alpha=0.2); axes[1].set_xlim(0, 112); axes[1].set_xticks(range(0, 101, 20))
    for yi, v in zip(y, cov): axes[1].text(102, yi, f"{v:.0f}%", va="center", fontsize=7)
    axes[1].set_title("(b) How often the 95%\ninterval covers the truth"); axes[1].set_xlabel("coverage (%)")
    # (c) honesty of the reported SE
    ratio = (t.loc[order, "Avg_SE"] / t.loc[order, "SD"])
    axes[2].barh(y, ratio, color=[FAMILY[m] for m in order]); axes[2].axvline(1, color="black", ls="--", lw=1.2)
    for yi, v in zip(y, ratio): axes[2].text(1.14, yi, f"{v:.2f}", va="center", fontsize=7)
    axes[2].set_xlim(0, 1.25); axes[2].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[2].set_title("(c) Reported SE ÷ true\nspread (1 = honest)"); axes[2].set_xlabel("ratio")
    fig.legend(handles=LEGEND, loc="lower center", ncol=4, frameon=False, fontsize=7)
    plt.tight_layout(rect=[0, 0.08, 1, 1]); plt.savefig(f"{ROOT}/figures/fig{n}_simulation_{mech}.png", dpi=200); plt.close()
for mech, n in zip(MECHS, [4, 5, 6]): mech_figure(mech, n)

# ---------- Fig 7: individual intervals under MAR for four key methods
fig, axes = plt.subplots(1, 4, figsize=(7.8, 3.4), sharex=True)
for ax, m in zip(axes, ["Complete Case", "Stochastic Regression", "MICE (normal)", "EM"]):
    t = sim[(sim.Mechanism == "MAR") & (sim.Method == m) & (sim.Rep < 40)]
    lo, hi = (t.Beta1 - 1.96 * t.SE).to_numpy(), (t.Beta1 + 1.96 * t.SE).to_numpy(); ok = (lo <= 2) & (2 <= hi)
    ax.hlines(t.Rep, lo, hi, colors=np.where(ok, "#2e8b57", "#c0392b"), lw=1.4)
    ax.plot(t.Beta1, t.Rep, "k.", ms=3); ax.axvline(2, color="black", ls="--"); ax.set_yticks([])
    ax.set_title(f"{m}\ncoverage: {s[(s.Mechanism=='MAR')&(s.Method==m)]['Coverage (%)'].iloc[0]:.0f}%", fontsize=8)
    ax.set_xlabel("β1 and interval")
plt.tight_layout(); plt.savefig(f"{ROOT}/figures/fig7_intervals_MAR.png", dpi=200); plt.close()

# ---------- Fig 8: at-a-glance heatmaps (bias and coverage)
def heat(ax, df, title, cmap, vmin, vmax, fmt, center_text):
    im = ax.imshow(df.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(MECHS); ax.set_yticks(range(len(df))); ax.set_yticklabels(df.index); ax.set_title(title)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.iloc[i, j]; rgba = im.cmap(im.norm(v)); lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, ("0.00" if (fmt.startswith("{:+") and abs(v) < 0.005) else fmt.format(v)), ha="center", va="center", color="black" if lum > 0.5 else "white", fontsize=7.5)
    for sp in ax.spines.values(): sp.set_visible(False)
    return im
bias = s.pivot(index="Method", columns="Mechanism", values="Bias").loc[METHODS, MECHS]
cov = s.pivot(index="Method", columns="Mechanism", values="Coverage (%)").loc[METHODS, MECHS]
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.5))
im1 = heat(axes[0], bias, "Bias of β1 (0 = perfect)", "RdBu_r", -0.6, 0.6, "{:+.2f}", True)
im2 = heat(axes[1], cov, "Coverage of 95% interval (95 = perfect)", "RdYlGn", 0, 100, "{:.0f}%", False)
axes[1].set_yticklabels([])
plt.tight_layout(); plt.savefig(f"{ROOT}/figures/fig8_at_a_glance.png", dpi=200); plt.close()
print("figures:", sorted(os.listdir(f"{ROOT}/figures")))
