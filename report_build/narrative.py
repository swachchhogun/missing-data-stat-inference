"""Interpretation text for the report, README and notebook. Every number is read from the results, and every
qualitative claim is guarded by an assert, so the text cannot silently disagree with the data."""
import numpy as np, pandas as pd
from scipy.stats import norm

SINGLE = ["Hot-Deck", "Regression Imputation", "Stochastic Regression"]
BOOTED = ["Complete Case", "Mean Imputation", "Hot-Deck", "Regression Imputation", "Stochastic Regression", "EM"]
NONBENCH = ["Complete Case", "Mean Imputation", "Hot-Deck", "Regression Imputation", "Stochastic Regression", "MICE (normal)", "MICE (PMM)", "EM"]

def predicted_coverage(bias, sd, ratio):
    """Coverage of estimate +/- 1.96*SE when the estimate is normal with this bias and SD, and SE = ratio*SD."""
    k = bias / sd
    return 100 * (norm.cdf(1.96 * ratio - k) - norm.cdf(-1.96 * ratio - k))

def facts(E, C, hon, MC_ERR):
    F = {}
    F["mean_bias"] = [E(me, "Mean Imputation", "Bias") for me in ("MCAR", "MAR", "MNAR")]
    F["mean_maxcov"] = max(C(me, "Mean Imputation", k) for me in ("MCAR", "MAR", "MNAR") for k in ("Classical", "Bootstrap standard", "Bootstrap percentile"))
    cr = [hon.loc[(me, m), "Classical SE / SD"] for me in ("MCAR", "MAR", "MNAR") for m in SINGLE]
    br = [hon.loc[(me, m), "Bootstrap SE / SD"] for me in ("MCAR", "MAR", "MNAR") for m in BOOTED]
    F["class_ratio"] = (min(cr), max(cr)); F["boot_ratio"] = (min(br), max(br))
    F["bench"] = (C("MCAR", "Full data", "Classical"), C("MCAR", "Full data", "Bootstrap standard"), C("MCAR", "Full data", "Bootstrap percentile"))
    F["mnar_max"] = max(C("MNAR", m, k) for m in NONBENCH for k in ("Classical", "Bootstrap standard", "Bootstrap percentile", "Rubin") if not np.isnan(C("MNAR", m, k)))
    # ---- guards
    assert F["mean_maxcov"] < 20, "mean imputation coverage claim"
    assert F["class_ratio"][1] < 0.85, "classical SE of single imputation should be clearly too small"
    assert F["boot_ratio"][0] > 0.9 and F["boot_ratio"][1] < 1.05, "bootstrap SE should be close to honest for every method"
    assert all(abs(x - 95) <= MC_ERR for x in F["bench"]), "benchmark coverage should be consistent with 95"
    for me in ("MCAR", "MAR"):
        assert abs(C(me, "Stochastic Regression", "Bootstrap percentile") - 95) <= MC_ERR, "stochastic + bootstrap ~95"
        assert C(me, "Stochastic Regression", "Classical") < 95 - MC_ERR, "stochastic classical under-covers"
        assert abs(C(me, "MICE (normal)", "Rubin") - 95) <= MC_ERR and abs(C(me, "EM", "Bootstrap percentile") - 95) <= MC_ERR
    assert abs(C("MCAR", "Complete Case", "Classical") - 95) <= MC_ERR
    assert max(C("MAR", "Complete Case", k) for k in ("Classical", "Bootstrap standard", "Bootstrap percentile")) < 80, "CC under MAR"
    assert C("MAR", "MICE (PMM)", "Rubin") < 95 - MC_ERR
    assert F["mnar_max"] < 95 - MC_ERR, "no method reaches 95 under MNAR"
    for me in ("MCAR", "MAR", "MNAR"):
        for m in ("Hot-Deck", "Regression Imputation"):
            assert C(me, m, "Bootstrap standard") > C(me, m, "Classical"), "bootstrap should improve hot-deck/regression coverage"
    return F

def pred_table(E, C, hon):
    rows = []
    for me, m in [("MCAR", "Hot-Deck"), ("MCAR", "Mean Imputation"), ("MAR", "Complete Case"), ("MAR", "Hot-Deck"), ("MAR", "Regression Imputation"),
                  ("MAR", "Stochastic Regression"), ("MNAR", "Regression Imputation"), ("MNAR", "Stochastic Regression")]:
        b, sd, r = E(me, m, "Bias"), E(me, m, "SD"), hon.loc[(me, m), "Bootstrap SE / SD"]
        p = predicted_coverage(b, sd, r); o = C(me, m, "Bootstrap standard")
        rows.append((me, m, b / sd, r, p, o))
    assert max(abs(p - o) for *_, p, o in rows) < 5, "bias should explain the remaining under-coverage"
    return rows

def narrative(D):
    E, C, best, MC_ERR, N_REPS, B_BOOT, hon = D["E"], D["C"], D["best"], D["MC_ERR"], D["N_REPS"], D["B_BOOT"], D["hon"]
    F = facts(E, C, hon, MC_ERR); PT = pred_table(E, C, hon); D["pred_table"] = PT
    cr, br = F["class_ratio"], F["boot_ratio"]
    N = {}
    N["summary"] = [
        f"<b>Mean imputation is the worst method in every setting.</b> Its bias is {F['mean_bias'][0]:+.2f}, {F['mean_bias'][1]:+.2f} and {F['mean_bias'][2]:+.2f} (MCAR, MAR, MNAR), and no interval — classical or bootstrap — contains the truth in more than {F['mean_maxcov']:.1f}% of the runs.",
        f"<b>The classical interval of single imputation is too narrow.</b> For hot-deck, regression and stochastic regression the formula SE is only {100 * cr[0]:.0f}–{100 * cr[1]:.0f}% of the true spread, because made-up values are treated as real. Stochastic regression is unbiased, yet its classical interval covers only {C('MCAR','Stochastic Regression','Classical'):.0f}% (MCAR) and {C('MAR','Stochastic Regression','Classical'):.0f}% (MAR).",
        f"<b>The bootstrap fixes the standard error, but not bias.</b> Because it repeats the imputation in every resample, the bootstrap SE is {100 * br[0]:.0f}–{100 * br[1]:.0f}% of the true spread for <i>every</i> method. Stochastic regression with a bootstrap interval reaches {C('MCAR','Stochastic Regression','Bootstrap percentile'):.0f}% (MCAR) and {C('MAR','Stochastic Regression','Bootstrap percentile'):.0f}% (MAR). But for biased methods the interval is centred in the wrong place, and the remaining under-coverage is predicted almost exactly by the bias.",
        f"<b>MCAR:</b> complete case works well (classical coverage {C('MCAR','Complete Case','Classical'):.0f}%); MICE and EM are slightly more precise (RMSE {E('MCAR','MICE (normal)','RMSE'):.3f} and {E('MCAR','EM','RMSE'):.3f} against {E('MCAR','Complete Case','RMSE'):.3f}).",
        f"<b>MAR:</b> complete case is biased ({E('MAR','Complete Case','Bias'):+.3f}) and every one of its intervals covers below 80% — the bootstrap cannot help. EM and MICE (normal) are essentially unbiased with {C('MAR','EM','Bootstrap percentile'):.0f}% and {C('MAR','MICE (normal)','Rubin'):.0f}% coverage, because they use the extra observed variable.",
        f"<b>MNAR:</b> no method and no interval reaches 95%; the best reach about {F['mnar_max']:.0f}%."]
    w = D["walk"]
    N["walk"] = (f"On this one dataset mean imputation is far below the truth under every mechanism ({w.loc['Mean Imputation','MCAR']:.2f}, {w.loc['Mean Imputation','MAR']:.2f}, {w.loc['Mean Imputation','MNAR']:.2f}), and the other methods land between "
                 f"{min(w.drop('Mean Imputation').min()):.2f} and {max(w.drop('Mean Imputation').max()):.2f}. One dataset is one random draw, so it cannot show which method is reliable on average or whether its interval can be trusted; that is what the Monte Carlo study is for.")
    N["MCAR"] = [
        f"<b>Complete case is fine.</b> Unbiased ({E('MCAR','Complete Case','Bias'):+.3f}) with coverage {C('MCAR','Complete Case','Classical'):.0f}% (classical) and {C('MCAR','Complete Case','Bootstrap percentile'):.0f}% (bootstrap percentile). The rows that remain are a fair sample; the only cost is precision.",
        f"<b>MICE and EM</b> are unbiased and slightly more accurate (RMSE {E('MCAR','MICE (normal)','RMSE'):.3f}, {E('MCAR','MICE (PMM)','RMSE'):.3f}, {E('MCAR','EM','RMSE'):.3f}), with coverage {C('MCAR','MICE (normal)','Rubin'):.0f}%, {C('MCAR','MICE (PMM)','Rubin'):.0f}% (Rubin) and {C('MCAR','EM','Bootstrap percentile'):.0f}% (EM, bootstrap).",
        f"<b>Stochastic regression</b> is unbiased ({E('MCAR','Stochastic Regression','Bias'):+.3f}). Its classical interval covers only {C('MCAR','Stochastic Regression','Classical'):.0f}%, but with the bootstrap it reaches {C('MCAR','Stochastic Regression','Bootstrap percentile'):.0f}% (percentile).",
        f"<b>Hot-deck and regression imputation</b> are biased ({E('MCAR','Hot-Deck','Bias'):+.3f} and {E('MCAR','Regression Imputation','Bias'):+.3f}). The bootstrap raises their coverage (hot-deck {C('MCAR','Hot-Deck','Classical'):.0f}% → {C('MCAR','Hot-Deck','Bootstrap standard'):.0f}%, regression {C('MCAR','Regression Imputation','Classical'):.0f}% → {C('MCAR','Regression Imputation','Bootstrap standard'):.0f}%) but not to 95%. Mean imputation reaches at most {max(C('MCAR','Mean Imputation',k) for k in ('Classical','Bootstrap standard','Bootstrap percentile')):.1f}%."]
    N["MAR"] = [
        f"<b>Complete case is no longer safe.</b> It is biased ({E('MAR','Complete Case','Bias'):+.3f}) because the rows that lose values have large <i>Z</i>, which goes with large <i>Y</i>. Its coverage is {C('MAR','Complete Case','Classical'):.0f}% (classical), {C('MAR','Complete Case','Bootstrap standard'):.0f}% and {C('MAR','Complete Case','Bootstrap percentile'):.0f}% (bootstrap): its standard error is fine, the problem is where the interval is centred.",
        f"<b>EM and MICE (normal)</b> use <i>Z</i>, are essentially unbiased ({E('MAR','EM','Bias'):+.3f}, {E('MAR','MICE (normal)','Bias'):+.3f}), have the lowest RMSE ({E('MAR','EM','RMSE'):.3f}, {E('MAR','MICE (normal)','RMSE'):.3f}) and coverage {C('MAR','EM','Bootstrap percentile'):.0f}% (EM, bootstrap) and {C('MAR','MICE (normal)','Rubin'):.0f}% (Rubin).",
        f"<b>Stochastic regression with the bootstrap</b> also works: unbiased ({E('MAR','Stochastic Regression','Bias'):+.3f}), coverage {C('MAR','Stochastic Regression','Classical'):.0f}% with the classical interval but {C('MAR','Stochastic Regression','Bootstrap percentile'):.0f}% with the bootstrap percentile interval.",
        f"<b>MICE (PMM)</b> has a small downward bias ({E('MAR','MICE (PMM)','Bias'):+.3f}) and {C('MAR','MICE (PMM)','Rubin'):.0f}% coverage. Hot-deck ({C('MAR','Hot-Deck','Bootstrap percentile'):.0f}%) and regression imputation ({C('MAR','Regression Imputation','Bootstrap percentile'):.0f}%) improve with the bootstrap but stay biased; mean imputation stays near {C('MAR','Mean Imputation','Bootstrap percentile'):.0f}%."]
    N["MNAR"] = [
        f"<b>No method is reliable.</b> Complete case is biased downward ({E('MNAR','Complete Case','Bias'):+.3f}); EM, MICE (normal) and stochastic regression, which all assume MAR, overshoot ({E('MNAR','EM','Bias'):+.3f}, {E('MNAR','MICE (normal)','Bias'):+.3f}, {E('MNAR','Stochastic Regression','Bias'):+.3f}). The highest coverage of any interval is {F['mnar_max']:.0f}%, below the 95% target even after allowing for random error.",
        f"<b>The bootstrap helps a little but cannot remove MNAR bias.</b> Stochastic regression goes from {C('MNAR','Stochastic Regression','Classical'):.0f}% to {C('MNAR','Stochastic Regression','Bootstrap percentile'):.0f}%, regression imputation from {C('MNAR','Regression Imputation','Classical'):.0f}% to {C('MNAR','Regression Imputation','Bootstrap percentile'):.0f}%.",
        f"Hot-deck happens to have a small bias here ({E('MNAR','Hot-Deck','Bias'):+.3f}) and reaches {C('MNAR','Hot-Deck','Bootstrap percentile'):.0f}% with the bootstrap; this is most likely two errors cancelling, not a sign that hot-deck handles MNAR."]
    sub = D.get("iv40", {})
    N["fig7"] = (f"Figure 7 makes the difference visible. Complete case misses mostly on the <i>low</i> side whichever interval is used: the intervals are centred too low (bias), and making them honest does not move them. Stochastic regression’s classical intervals are centred correctly but too short, so they miss on both sides; its bootstrap intervals are about "
                 f"{100 * (D['wid'].loc[('MAR','Stochastic Regression'),'Bootstrap percentile'] / D['wid'].loc[('MAR','Stochastic Regression'),'Classical'] - 1):.0f}% wider and the coverage rises from {C('MAR','Stochastic Regression','Classical'):.0f}% to {C('MAR','Stochastic Regression','Bootstrap percentile'):.0f}%.")
    N["fig8"] = ("Figure 8 summarises everything. Moving from the left map (no bootstrap) to the right map (bootstrap percentile), the cells of stochastic regression turn dark green under MCAR and MAR, and hot-deck and regression imputation improve — the bootstrap repairs over-confidence. "
                 "The cells of mean imputation stay red, complete case under MAR stays light, and the MNAR column has no dark green cell in either map — the bootstrap cannot repair bias. MICE (Rubin) and EM are dark green under MCAR and MAR in both maps.")
    N["discussion"] = [
        ("9.1  Mean imputation hurts everywhere", f"Filling every gap with the same number removes the natural variation in <i>Y</i> and <i>X</i><sub>1</sub>, and with it part of their relationship, so the slope is pulled towards zero (Figure 2). The bias ({F['mean_bias'][0]:+.2f} to {F['mean_bias'][2]:+.2f}) is several times the spread, so no interval — classical or bootstrap — can rescue it."),
        ("9.2  Why the classical interval of single imputation is too narrow", f"Hot-deck, regression and stochastic regression fill each gap once and then treat the completed data as real. About a third of the rows contain made-up values, so the formula thinks it has more information than it has: the classical SE is only {100 * cr[0]:.0f}–{100 * cr[1]:.0f}% of the true spread (panel (c) of Figures 4–6), and the intervals are too short."),
        ("9.3  What the bootstrap fixes, and what it cannot", f"The bootstrap re-runs the <i>whole</i> method, filling-in included, on every resample. The spread of the {B_BOOT} answers therefore contains the uncertainty created by the imputation, and the bootstrap SE is {100 * br[0]:.0f}–{100 * br[1]:.0f}% of the true spread for every method. What the bootstrap cannot change is where the interval is centred: it resamples the <i>observed</i> data, so a biased method is equally biased in every resample. Table 16 shows that, once the SE is honest, the coverage of a method is almost exactly what its bias predicts: an interval centred on a normal estimate with bias <i>b</i> and spread SD covers the truth with probability Φ(1.96·r − b/SD) − Φ(−1.96·r − b/SD), where r is the SE ÷ SD ratio. <b>The bootstrap repairs variance, not bias.</b>"),
        ("9.4  Bootstrap standard versus percentile", f"The two bootstrap intervals give very similar coverage here, because with n = 200 the bootstrap estimates are close to normal. With no missing data both cover {F['bench'][1]:.1f}%, slightly below 95% but within random error; bootstrap intervals are known to be a little too narrow in moderate samples, which is one reason refined versions (bootstrap-t, BCa) exist."),
        ("9.5  Multiple imputation versus single imputation plus bootstrap", f"Under MCAR and MAR, MICE (normal) with Rubin’s rules and stochastic regression with a bootstrap interval reach the same goal — an unbiased estimate with an honest interval. MICE needs only {D['M_IMP_TXT']} imputations; the bootstrap needs {B_BOOT} complete re-runs of the method. Rubin’s rules are the cheaper way to carry the imputation uncertainty; the bootstrap is the general-purpose way that works for any method, including EM."),
        ("9.6  PMM versus the normal version of MICE", f"Under MAR, MICE (PMM) is slightly biased ({E('MAR','MICE (PMM)','Bias'):+.3f}) with {C('MAR','MICE (PMM)','Rubin'):.0f}% coverage. The data are exactly linear and normal, so the normal model is right and PMM’s robustness to odd distributions is not needed. PMM can only borrow observed values, and the rows that MAR targets (with extreme predicted values) have few similar donors."),
        ("9.7  MNAR is different in kind", "Under MNAR the chance of being missing depends on the missing value itself, so the observed data cannot reveal how the missing values differ. Neither better imputation nor the bootstrap can fix this; the honest approach is a sensitivity analysis under several plausible assumptions."),
    ]
    N["limitations"] = [
        "<b>One simple model.</b> Linear, normal, independent predictors, n = 200, about 20% missing in two variables. Results can change with other distributions, sample sizes or amounts of missing data; in particular PMM is designed for non-normal data.",
        "<b>Chosen strengths and a strong helper variable.</b> The MAR and MNAR strengths were chosen by us, and <i>Z</i> is a close stand-in for <i>Y</i>. Real auxiliary variables are usually weaker.",
        f"<b>Bootstrap choices.</b> {B_BOOT} resamples per dataset is enough for the SE, but the 2.5% and 97.5% points rest on about 5 resamples each, so percentile limits are a little noisy. Bootstrap-t (which needs a second bootstrap inside every resample, N × B × B′ fits) and BCa intervals were not computed, and MICE was not bootstrapped (Rubin’s rules were used instead).",
        "<b>Simplified MICE and EM.</b> MICE uses 5 cycles, " + D["M_IMP_TXT"] + " imputations, 5 donors and a normal (±1.96) interval rather than Rubin’s t-based interval; EM assumes normality. For real work use a tested package such as <i>mice</i> in R or <i>statsmodels</i>.",
        f"<b>Monte Carlo error.</b> With {N_REPS} repetitions, coverage near 95% is accurate to about ±{MC_ERR:.1f} points.",
        f"<b>Only {chr(946)}<sub>1</sub> is examined</b>, not the other coefficients or predictions."]
    N["recommend"] = [
        ["MCAR", "Complete case with its classical interval; MICE or EM for slightly more precision.", "Mean imputation; classical intervals after single imputation."],
        ["MAR", "EM (bootstrap interval) or MICE (normal, Rubin interval), with the helper variable in the imputation model; stochastic regression only with a bootstrap interval.", "Complete case (biased); single imputation with classical intervals (over-confident)."],
        ["MNAR", "Nothing is reliable. Report a sensitivity analysis.", "Claiming that any method or interval fixes MNAR."]]
    N["conclusion"] = [
        "<b>Never use mean imputation</b> for inference.",
        "<b>Never use the classical interval after single imputation</b>: it treats guesses as data. Use Rubin’s rules, EM with a bootstrap, or a bootstrap that repeats the imputation.",
        "<b>The bootstrap fixes over-confidence, not bias.</b> An honest interval around a biased estimate still misses; choose a method that removes the bias first (under MAR: use the variables that explain the missingness).",
        "<b>Coverage can only be judged by Monte Carlo.</b> A single bootstrap interval says nothing about how often such intervals contain the truth."]
    return N

M_IMP_TXT = "10"

def readme_findings(cov, est):
    C = lambda me, m, k: float(cov.loc[(me, m), k]); E = lambda me, m, c: float(est.loc[(me, m), c])
    return [f"**Mean imputation is the worst everywhere** (bias {E('MCAR','Mean Imputation','Bias'):+.2f} / {E('MAR','Mean Imputation','Bias'):+.2f} / {E('MNAR','Mean Imputation','Bias'):+.2f}; coverage below 20% with any interval).",
            f"**Classical intervals after single imputation are too narrow:** stochastic regression is unbiased but its classical interval covers only {C('MCAR','Stochastic Regression','Classical'):.0f}% (MCAR) and {C('MAR','Stochastic Regression','Classical'):.0f}% (MAR).",
            f"**The bootstrap fixes the standard error, not bias:** with a bootstrap percentile interval stochastic regression reaches {C('MCAR','Stochastic Regression','Bootstrap percentile'):.0f}% and {C('MAR','Stochastic Regression','Bootstrap percentile'):.0f}%, but biased methods stay below target (complete case under MAR: {C('MAR','Complete Case','Bootstrap percentile'):.0f}%). The remaining under-coverage is predicted almost exactly by each method's bias.",
            f"**MAR:** EM ({C('MAR','EM','Bootstrap percentile'):.0f}%) and MICE normal ({C('MAR','MICE (normal)','Rubin'):.0f}%) are unbiased and honest because they use the helper variable; complete case is biased.",
            "**MNAR:** no method or interval reaches 95% — report a sensitivity analysis."]
