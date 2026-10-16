# Missing Data and Statistical Inference

> **Note:** this folder holds the **extended study** (200 observations, gaps in two variables, a helper variable, eight methods, 300 repetitions). The original single-dataset study (100 observations, only Y incomplete, seven methods) is in the repository root: `notebooks/`, `report/`, `figures/`, `results/`.

A simulation study that compares **eight ways of handling missing values** in a regression, separately for the three kinds of missingness (**MCAR, MAR and MNAR**).

The data are made up from a model where the true answer is known (`Y = 5 + 2·X1 − 1.5·X2 + X3 + error`, so the slope we care about is **β₁ = 2**). About 20% of the values of `Y` and `X1` are then deleted, and each method tries to recover the right answer *and* a trustworthy 95% confidence interval.

**Read the full report here:** [`report/Missing_Data_Statistical_Inference_Report.pdf`](report/Missing_Data_Statistical_Inference_Report.pdf)

## The question

> Within each type of missingness, which way of handling the missing values gives the most accurate estimate and the most honest confidence interval?

## The eight methods

Complete-case analysis · Mean imputation · Hot-deck · Regression imputation · Stochastic regression · MICE (normal) · MICE (PMM) · EM algorithm

## Main findings (300 simulated datasets per mechanism)

| Missingness | What works | What to avoid |
|---|---|---|
| **MCAR** (pure chance) | Complete case is fine. MICE or EM are slightly more precise. | Mean imputation and single imputation |
| **MAR** (depends on an observed variable) | **EM or MICE (normal)**, using the extra observed variable | Complete case (biased) and single imputation |
| **MNAR** (depends on the missing value) | Nothing is reliable. Report a sensitivity analysis. | Claiming any method "fixes" MNAR |

In every case:

* **Mean imputation is the worst method** everywhere.
* **Hot-deck, regression imputation and stochastic regression give confidence intervals that are too narrow**, because they treat made-up values as real data (their reported standard errors are only about 70–85% of the true spread).
* **MICE and EM report honest uncertainty** when the data are MCAR or MAR (coverage of the 95% interval is about 95%).

## What is in this repository

```
├── report/
│   └── Missing_Data_Statistical_Inference_Report.pdf   the full written report (21 pages)
├── notebook/
│   └── missing_data_simulation.ipynb                   all the code, step by step, with outputs
├── results/
│   ├── part_a_results.csv                              Part A: one dataset, every method
│   ├── simulation_summary.csv                          Part B: summary of the 300 repetitions
│   └── simulation_all_repetitions.csv                  Part B: every single run
├── figures/                                            the figures used in the report
├── report_build/                                       scripts that redraw the figures and rebuild the PDF
├── requirements.txt
└── README.md
```

## How the study is organised

* **Part A – one worked example.** One dataset, analysed step by step, with bootstrap standard errors.
* **Part B – simulation study.** The whole experiment is repeated 300 times with fresh data, so we can see how each method behaves *on average* (bias, spread, RMSE, and how often the 95% interval contains the truth).

All comparisons are made **within** each mechanism.

## How to run the notebook

1. Install the requirements: `pip install -r requirements.txt`
2. Open `notebook/missing_data_simulation.ipynb` in Jupyter (or upload it to Google Colab) and choose **Run all**.

The full run takes about **8 minutes**. For a quick test, set `N_REPS = 50` in the settings block near the top (Step 1b). Every code block starts from its own random seed, so any block can be re-run on its own and gives the same answer. The notebook also writes the three CSV files listed above.

## How to rebuild the report

```
python report_build/make_figures.py    # redraws the figures from the CSV files in results/
python report_build/build_pdf.py       # rebuilds the PDF from the notebook outputs and the figures
```

## Requirements

Python 3 with `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `matplotlib` (for the notebook) and `reportlab`, `Pillow` (for the report scripts). The project was developed and tested with numpy 2.4, pandas 3.0, scipy 1.17, scikit-learn 1.8, statsmodels 0.15, matplotlib 3.10 and reportlab 4.4.

## References

1. Dempster, A. P., Laird, N. M., and Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B*, 39(1), 1–38.
2. Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*. Wiley.
3. van Buuren, S. (2018). *Flexible Imputation of Missing Data*, 2nd ed. Chapman & Hall/CRC.
